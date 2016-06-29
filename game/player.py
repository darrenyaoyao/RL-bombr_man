"""A controller for the player's man"""

import pygame
import time
import random
import numpy as np
from keras.models import model_from_json

import serge.engine
import serge.sound

from theme import G
BOMBR_COLUMN = 19
BOMBR_ROW = 19

class Player(object):
    """Represents the player's control of the actor"""

    def __init__(self):
        """Initialise the controller"""
        self.keyboard = serge.engine.CurrentEngine().getKeyboard()
        self._last_move = time.time()
        self._move_interval = G('player-move-interval')
        self.walk = serge.sound.Sounds.getItem('walk')
        self.initialmodel()
        self.bomb = False

    def updateController(self, interval, man, board):
        """Update the control of the actor"""
        #
        # Find out the way the player wants to go
        current_state = board.observation[-1]#current state, no observation
        if len(board.observation) > 1:
            last_state = board.observation[-2]
        else:
            last_state = board.observation[-1]

        if board.options.random:
            direction = self.random_policy(current_state)
        elif board.options.supervised_policy:
            direction = self.supervised_policy(last_state)
        else:
            direction = None
            if self.keyboard.isDown(pygame.K_LEFT):
                direction = (-1, 0)
                current_state["action"] = 2
            if self.keyboard.isDown(pygame.K_RIGHT):
                direction = (+1, 0)
                current_state["action"] = 4
            if self.keyboard.isDown(pygame.K_UP):
                direction = (0, -1)
                current_state["action"] = 6
            if self.keyboard.isDown(pygame.K_DOWN):
                direction = (0, +1)
                current_state["action"] = 8

        #
        # Check that we can go there
        if direction and board.canMove(man, direction) and time.time() - self._last_move > self._move_interval:
            man.log.debug('Moving player by %s, %s' % direction)
            board.moveMan(man, direction)
            self.walk.play()
            self._last_move = time.time()
        #
        # See if any bombs should be dropped
        if board.options.random and direction == None:
            if man.canDropBomb():
                current_state["action"] += 1
                board.dropBomb(man)
                serge.sound.Sounds.play('drop')
        elif self.bomb:
            if man.canDropBomb():
                board.dropBomb(man)
                serge.sound.Sounds.play('drop')
                self.bomb = False
        elif self.keyboard.isClicked(pygame.K_SPACE):
            if man.canDropBomb():
                current_state["action"] += 1
                board.dropBomb(man)
                serge.sound.Sounds.play('drop')

    def initialmodel(self):
        self.model = model_from_json(open("./train/npyNmodel/model.json").read())
        self.model.load_weights("./train/npyNmodel/model_weight_classified.h5")
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def supervised_policy(self, agent_dic):
        if agent_dic["observation"] != []:
            obser = np.reshape( agent_dic["observation"], (BOMBR_ROW, BOMBR_COLUMN))
        else:
            obser = np.zeros((BOMBR_ROW, BOMBR_COLUMN))
        observation = np.zeros((1, BOMBR_ROW, BOMBR_COLUMN))
        observation[0] = obser
        action = self.model.predict_classes(observation)
        print (action)
        if action % 2 == 1:
            self.bomb = True
        else:
            self.bomb = False
        if action == 0 or action == 1:
            agent_dic["action"] = 0
            return (0, 0)
        elif action == 2 or action ==3:
            agent_dic["action"] = 2
            return (-1, 0)
        elif action == 4 or action ==5:
            agent_dic["action"] = 4
            return (+1, 0)
        elif action == 6 or action ==7:
            agent_dic["action"] = 6
            return (0, +1)
        elif action == 8 or action ==9:
            agent_dic["action"] = 8
            return (0, -1)
        else:
            return None


    def random_policy(self, agent_dic):
        a = random.randint(0,5)
        if a == 0:
            agent_dic["action"] = 0
            return (0, 0)
        elif a == 1:
            agent_dic["action"] = 2
            return (-1, 0)
        elif a == 2:
            agent_dic["action"] = 4
            return (+1, 0)
        elif a == 3:
            agent_dic["action"] = 6
            return (0, +1)
        elif a == 4:
            agent_dic["action"] = 8
            return (0, -1)
        else:
            agent_dic["action"] = 0
            return None
