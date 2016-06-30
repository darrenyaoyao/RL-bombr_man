import numpy as np
import sys

BOMBR_ROW = 19
BOMBR_COLUMN = 19
DEFAULT = int(15*(2**0.5))

class find_feature():
    def __init__(self):
        pass
    def find_distance(self, player_pos, obj_pos):
        d = ((obj_pos[0]-player_pos[0])**2 + (obj_pos[1]-player_pos[1])**2)**0.5
        return int(d)

    def find_nearest(self, player_pos, objects_pos):
        min_distance = DEFAULT
        for i in range(len(objects_pos)):
            d = ((objects_pos[i][0]-player_pos[0])**2 + (objects_pos[i][1]-player_pos[1])**2)**0.5
            if d < min_distance:
                min_distance = d
        return int(min_distance)

    def check_direction(self, player_pos, state):
        if state[player_pos[0]-1][player_pos[1]] == 4:
            return 0 #if can move up, return 0
        elif state[player_pos[0]+1][player_pos[1]] == 4:
            return 1 #if can move down, return 1
        elif state[player_pos[0]][player_pos[1]-1] == 4:
            return 2 #if can move left, return 2
        elif state[player_pos[0]][player_pos[1]+1] == 4:
            return 3 #if can move right, return 3
        else:
            return 4


    def find_player(self, player_pos, observation):
        count = 3
        while len(player_pos) == 0:
            for i in range(len(observation[-count])):
                for j in range(len(observation[-count][i])):
                    if observation[-count][i][j] == 6:
                        player_pos.append(i)
                        player_pos.append(j)
                        break
            count += 1

    def find_ai(self, ai_pos, observation):
        count = 3
        while len(ai_pos) == 0:
            for i in range(len(observation[-count])):
                for j in range(len(observation[-count][i])):
                    if observation[-count][i][j] == 7:
                        ai_pos.append(i)
                        ai_pos.append(j)
                        break
            count += 1

    def parse_feature(self, observation, obser):
        s = obser
        default_feature = [DEFAULT, DEFAULT, DEFAULT, DEFAULT, DEFAULT, 4]
        if (s == np.zeros((BOMBR_ROW, BOMBR_COLUMN))).all():
            return np.asarray(default_feature)
        features = []
        player_pos = []
        ai_pos = []
        flag_pos = []
        for i in range(len(s)):
            for j in range(len(s[i])):
                if s[i][j] == 6:
                    player_pos.append(i)
                    player_pos.append(j)
                elif s[i][j] == 7:
                    ai_pos.append(i)
                    ai_pos.append(j)
                elif s[i][j] == 0:
                    tmp = []
                    tmp.append(i)
                    tmp.append(j)
                    flag_pos.append(tmp)
        if len(player_pos) == 0:
            self.find_player(player_pos, observation)
        if len(ai_pos) == 0:
            self.find_ai(ai_pos, observation)
        box_pos = []
        bomb_pos = []
        explosion_pos = []
        for i in range(player_pos[0]-2, player_pos[0]+2):
            for j in range(player_pos[1]-2, player_pos[1]+2):
                if s[i][j] == 2:
                    tmp = []
                    tmp.append(i)
                    tmp.append(j)
                    box_pos.append(tmp)
                elif s[i][j] == 8:
                    tmp = []
                    tmp.append(i)
                    tmp.append(j)
                    bomb_pos.append(tmp)
                elif s[i][j] == 9:
                    tmp = []
                    tmp.append(i)
                    tmp.append(j)
                    explosion_pos.append(tmp)
        features.append(self.find_distance(player_pos, ai_pos))
        features.append(self.find_nearest(player_pos, flag_pos)) #if no flag, distance = 19*(2**0.5)
        features.append(self.find_nearest(player_pos, box_pos)) #if no box in 5*5, distance = 19*(2**0.5)
        features.append(self.find_nearest(player_pos, bomb_pos)) #if no bomb in 5*5, distance = 19*(2**0.5)
        features.append(self.find_nearest(player_pos, explosion_pos)) #if no explosion in 5*5, distance = 19*(2**0.5)
        features.append(self.check_direction(player_pos, s))
        return features
