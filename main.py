import io 
import re 
import cv2 

import zmq 
import json 

import httpx
import asyncio
import subprocess
import pyaudio

import numpy as np 

import gtts
import openai

import click 
import operator as op 

from async_timeout import timeout 
from loguru import logger 

from rich.console import Console
from enum import Enum

from typing import List 

def draw_grid(table:np.ndarray, hx:int, wx:int, states:List[List[int]]):
    map_1 = {
        1: 'X',
        2: 'O'
    }

    map_2 = {
        1: (0, 0, 255),
        2: (255, 0, 0)
    }
    

    i = 0
    for row in states:
        j = 0
        for col in row:
            if col != 0:
                a = i * hx + hx // 2 + hx // 4
                b = j * wx + wx // 2 - wx // 4  
                cv2.putText(
                    table,
                    map_1[col],
                    (b, a),
                    cv2.FONT_HERSHEY_COMPLEX,
                    5,
                    map_2[col],
                    3
                ) 
            j += 1
        i += 1 


def playsound(player, audiostream:bytes): 
    stream = player.open(format=pyaudio.paInt16, channels=2, rate=44100, output=True)
    data = audiostream.read(1024)

    while len(data) > 0:
        stream.write(data)
        data = audiostream.read(1024)

    stream.stop_stream()
    stream.close()

class AgentRole(str, Enum):
    SERVER:str='SERVER'
    CLIENT:str='CLIENT'

class GPTAgent:
    def __init__(self, name:str, role:AgentRole, address:str):
        self.name = name 
        self.role = role 
        self.address = address
        self.connection_type = {
            AgentRole.SERVER:'bind',
            AgentRole.CLIENT:'connect'
        }
        self.accents = {
            AgentRole.SERVER:'co.za',
            AgentRole.CLIENT:'co.uk'
        }
        self.agent2player = {
            AgentRole.SERVER: 2,
            AgentRole.CLIENT: 1
        }
        self.initialized = 0 

    def chat(self):
        console = Console()
        player = self.agent2player[self.role]
        memories = [
            {
                "role": "system", 
                "content": f"""
                    Ton rôle sera de jouer au tic tac toe avec un autre robot, tu es le player {player}, tu dois placer le pion {player} dans les cases libres. 
                    Une case libre est marquée par le nombre 0. Le player 1 va placer le pion 1 et la player 2 va placer le pion 2. 
                    La grille est un json avec la clé message(explication de ton choix) et cells(qui contient la matrice de taille 3x3).
                    Voici un exemple : 
                    {{'message': 'je place mon pion sur la première cellule', 'cells': [[1, 0, 0], [0, 0, ], [0, 0, 2]]}}
                    Sur cette grille, le player 1 a placé son pion sur la première case, tandis que le player 2 a placé son pion sur la dernière case. 
                    A chaque fois que tu reçois une grille, tu dois retourner la nouvelle grille en mettant ton pion {player} sur la grille. 
                    Le JSON doit être valide!
                    Ton choix doit être entre 0 et 8 en fonction de la disponiblité des cases.
                    Tu dois maximiser tes chances de gagner le jeu! 
                    Ta réponse doit être juste composée du JSON. Pas de commentaire, sauf si c'est la fin du JEU.
                """
            },  
        ]
        if self.role == AgentRole.CLIENT:
            grid = np.zeros(9, dtype=np.uint8)
            position = np.random.randint(0, 10)
            grid[position] = player
            grid = np.reshape(grid, (3, 3)).tolist()
            grid_2d = json.dumps(
                {
                    'message': f'Je place mon pion sur la case {position} pour commencer la partie',
                    'cells': grid 
                }
            )

            self.socket.send_json({
                'role': 'user', 
                'content': grid_2d  
            })

            console.print(grid_2d, style='white')
        
        W, H = 600, 600
        table = np.ones((H, W, 3), dtype=np.uint8) * 255
        wx = W // 3
        hx = H // 3 
        for i in range(0, H, hx):
            for j in range(0, W, wx):
                cv2.rectangle(table, (i, j), (i + hx, j + wx), 0, 5)
            
        pattern = r'{[^}]*}'

        if self.role == AgentRole.SERVER:
            cv2.imshow('000', table)
            cv2.waitKey(25)
            
        keep_loop = True 
        while keep_loop:
                
            try:
                socket_status = self.socket.poll(timeout=1000)
                if socket_status == zmq.POLLIN:
                    message = self.socket.recv_json()

                    text = message['content']  # from a user 
                    console.print(text, style='white')
                    status = re.search(pattern, text)
                    if status is not None:
                        json_string = status.group(0)
                        print('extracted json', json_string)
                        
                        if self.role == AgentRole.SERVER:
                            obj_ = json.loads(json_string)
                            draw_grid(table, hx, wx, obj_['cells'])
                            cv2.imshow('000', table)
                            cv2.waitKey(1000)

                        memories.append({
                            'role': 'user',
                            'content': json_string
                        })
                         
                        response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=memories 
                        )
                        
                        content = response['choices'][0]['message']['content']
                        self.socket.send_json({
                            'role': 'user',
                            'content': content
                        })
                        status = re.search(pattern, content)
                        if status is not None:
                            json_string = status.group(0)
                            memories.append(  # add assistant
                                {
                                    'role': 'assistant',
                                    'content': json_string 
                                }
                            )

                            if self.role == AgentRole.SERVER:
                                obj_ = json.loads(json_string)
                                draw_grid(table, hx, wx, obj_['cells'])
                                cv2.imshow('000', table)
                                cv2.waitKey(1000)

                        else:
                            keep_loop = False

                        
                    else:
                        logger.error('impossible to extract the json')
                        keep_loop = False 
            except KeyboardInterrupt:
                keep_loop = False 
            except Exception as e:
                logger.error(e) 
    
    def __enter__(self):
        try:
            self.ctx = zmq.Context()
            self.socket:zmq.Socket = self.ctx.socket(zmq.PAIR)
            op.attrgetter(self.connection_type[self.role])(self.socket)(self.address)
            self.initialized = 1
            logger.success(f'agent : {self.name} connected')
            self.player = pyaudio.PyAudio()
        except Exception as e:
            logger.error(e)
        return self 
    
    def __exit__(self, exc, val, taceback):
        if self.initialized == 1:
            self.socket.close(linger=0)
            self.ctx.term()
            logger.debug(f'agent {self.name} has released all zeromq ressources')
        self.player.terminate()
        logger.success(f'agent : {self.name} disconnected')


@click.command()
@click.option('--api_key', envvar='OPENAI_API_KEY', required=True)
@click.option('--name', help='name of the agent', type=str)
@click.option('--role', help='role is either SERVER or CLIENT', type=click.Choice([AgentRole.SERVER, AgentRole.CLIENT]))
@click.option('--address', help='zeromq address', type=str)
def start_conversation(api_key:str, name:str, role:AgentRole, address:str):
    openai.api_key = api_key 
    with GPTAgent(name, role, address) as agent:
        agent.chat()

if __name__ == '__main__':
    start_conversation()
