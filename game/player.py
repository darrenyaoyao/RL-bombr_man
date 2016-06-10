"""A controller for the player's man"""

import pygame
import time
import random

import serge.engine
import serge.sound

from theme import G


class Player(object):
    """Represents the player's control of the actor"""

    def __init__(self):
        """Initialise the controller"""
        self.keyboard = serge.engine.CurrentEngine().getKeyboard()
        self._last_move = time.time()
        self._move_interval = G('player-move-interval')
        self.walk = serge.sound.Sounds.getItem('walk')

    def updateController(self, interval, man, board):
        """Update the control of the actor"""
        #
        # Find out the way the player wants to go
        agent_dic = board.observation[-1]
        if board.options.random:
            direction = self.random_policy(agent_dic)
        else:
            direction = None        
            if self.keyboard.isDown(pygame.K_LEFT):
                direction = (-1, 0)
                agent_dic["action"] = 1
            if self.keyboard.isDown(pygame.K_RIGHT):
                direction = (+1, 0)
                agent_dic["action"] = 2
            if self.keyboard.isDown(pygame.K_UP):
                direction = (0, -1)
                agent_dic["action"] = 3
            if self.keyboard.isDown(pygame.K_DOWN):
                direction = (0, +1)
                agent_dic["action"] = 4
        
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
                agent_dic["action"] = 5
                board.dropBomb(man)
                serge.sound.Sounds.play('drop')
        elif self.keyboard.isClicked(pygame.K_SPACE):
            if man.canDropBomb():
                agent_dic["action"] = 5
                board.dropBomb(man)
                serge.sound.Sounds.play('drop')

    def random_policy(self, agent_dic):
        a = random.randint(0,5)
        if a == 0:
            agent_dic["action"] = 0
            return (0, 0)
        elif a == 1:
            agent_dic["action"] = 1
            return (-1, 0)
        elif a == 2:
            agent_dic["action"] = 2
            return (+1, 0)
        elif a == 3:
            agent_dic["action"] = 3
            return (0, -1)
        elif a == 4:
            agent_dic["action"] = 4
            return (0, +1)
        else:
            agent_dic["action"] = 5
            return None
