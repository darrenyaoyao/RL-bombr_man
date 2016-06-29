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

class PlayerAI(object):
    """Represents the player's control of the actor"""

    def __init__(self):
        """Initialise the controller"""
        self._last_move = time.time()
        self._move_interval = G('player-move-interval')
        self.walk = serge.sound.Sounds.getItem('walk')
        self.initialmodel()
        self.all_action = list()
        for i in range(10):
            action = np.zeros(10)
            action[i] = 1
            self.all_action.append(action)
        self.all_action = np.array(self.all_action)
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
            direction = self.random_policy()
        elif board.options.supervised_policy:
            direction = self.supervised_policy(last_state)
        elif board.options.dqn:
            direction = self.dqn_policy(last_state)

        if direction == (-1, 0):
            current_state["action"] = 2
        if direction == (+1, 0):
            current_state["action"] = 4
        if direction == (0, +1):
            current_state["action"] = 6
        if direction == (0, -1):
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
                current_state["action"] += 1
                board.dropBomb(man)
                serge.sound.Sounds.play('drop')
                self.bomb = False

    def initialmodel(self):
        self.model = model_from_json(open("./train/model.json").read())
        self.model.load_weights("./train/model_weight.h5")
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.dqnmodel = model_from_json(open("dqnmodel.json").read())
        self.dqnmodel.load_weights("dqnmodel_weight.h5")
        self.dqnmodel.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    def supervised_policy(self, agent_dic):
        if agent_dic["observation"] != []:
            obser = np.reshape( agent_dic["observation"], (BOMBR_ROW, BOMBR_COLUMN))
        else:
            obser = np.zeros((BOMBR_ROW, BOMBR_COLUMN))
        observation = np.zeros((1, BOMBR_ROW, BOMBR_COLUMN))
        observation[0] = obser
        action = self.model.predict_classes(observation)
        return self.action2direction(action, agent_dic)

    def dqn_policy(self, agent_dic):
        if agent_dic["observation"] != []:
            obser = np.reshape( agent_dic["observation"], (BOMBR_ROW, BOMBR_COLUMN))
        else:
            obser = np.zeros((BOMBR_ROW, BOMBR_COLUMN))
        observation = np.zeros((10, BOMBR_ROW, BOMBR_COLUMN))
        for i in range(10):
            observation[i] = obser
        Q = self.dqnmodel.predict([observation, self.all_action])
        maxQ = Q[0][0]
        action = 0
        for i in range(10):
            if Q[i][0] > maxQ:
                maxQ = Q[i][0]
                action = i
        return self.action2direction(action, agent_dic)

    def action2direction(self, action, agent_dic):
        if action % 2 == 1:
            self.bomb = True
        else:
            self.bomb = False
        if action == 0 or action == 1:
            return (0, 0)
        elif action == 2 or action ==3:
            return (-1, 0)
        elif action == 4 or action ==5:
            return (+1, 0)
        elif action == 6 or action ==7:
            return (0, +1)
        elif action == 8 or action ==9:
            return (0, -1)
        else:
            return None


    def random_policy(self, agent_dic):
        a = random.randint(0,5)
        if a == 0:
            return (0, 0)
        elif a == 1:
            return (-1, 0)
        elif a == 2:
            return (+1, 0)
        elif a == 3:
            return (0, +1)
        elif a == 4:
            return (0, -1)
        else:
            return None
