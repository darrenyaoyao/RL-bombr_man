import numpy as np
import sys

def find_distance(player_pos, obj_pos):
    d = 19*(2**0.5)
    if len(obj_pos) != 0:
        d = ((obj_pos[0]-player_pos[0])**2 + (obj_pos[1]-player_pos[1])**2)**0.5
    return d

def find_nearest(player_pos, objects_pos):
    min_distance = 19 * (2**0.5)
    for i in range(len(objects_pos)):
        d = ((objects_pos[i][0]-player_pos[0])**2 + (objects_pos[i][1]-player_pos[1])**2)**0.5
        if d < min_distance:
            min_distance = d
    return min_distance

def check_direction(player_pos, state):
    if state[player_pos[0]-1][player_pos[1]] == 4:
        return 0 #if can move up, return 0
    if state[player_pos[0]+1][player_pos[1]] == 4:
        return 1 #if can move down, return 1
    if state[player_pos[0]][player_pos[1]-1] == 4:
        return 2 #if can move left, return 2
    if state[player_pos[0]][player_pos[1]+1] == 4:
        return 3 #if can move right, return 3

def find_player(player_pos, states, k):
    count = 1
    while len(player_pos) == 0:
        for i in range(len(states[k-count])):
            for j in range(len(states[k-count][i])):
                if states[k-count][i][j] == 6:
                    player_pos.append(i)
                    player_pos.append(j)
                    break
        count += 1

def find_ai(ai_pos, states, k):
    count = 1
    while len(ai_pos) == 0:
        for i in range(len(states[k-count])):
            for j in range(len(states[k-count][i])):
                if states[k-count][i][j] == 7:
                    ai_pos.append(i)
                    ai_pos.append(j)
                    break
        count += 1

states = np.load(sys.argv[1])
features = []
for k, s in enumerate(states):
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
        find_player(player_pos, states, k)
    if len(ai_pos) == 0:
        find_ai(ai_pos, states, k)
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
    features_tmp = []
    features_tmp.append(find_distance(player_pos, ai_pos))
    features_tmp.append(find_nearest(player_pos, flag_pos)) #if no flag, distance = 19*(2**0.5)
    features_tmp.append(find_nearest(player_pos, box_pos)) #if no box in 5*5, distance = 19*(2**0.5)
    features_tmp.append(find_nearest(player_pos, bomb_pos)) #if no bomb in 5*5, distance = 19*(2**0.5)
    features_tmp.append(find_nearest(player_pos, explosion_pos)) #if no explosion in 5*5, distance = 19*(2**0.5)
    features_tmp.append(check_direction(player_pos, s))
    features.append(features_tmp)
features = np.asarray(features)
np.save(sys.argv[3], features)
