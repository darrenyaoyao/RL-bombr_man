import numpy as np
import random
BOMBR_COLUMN = 19
BOMBR_ROW = 19
FINALSTATE = np.full((15,15),3.0)
FINALACTION = np.full(10,1.0)
REWARD = 0
#ACTION_PERCENT_RETAIN = 0.2

class parseData:
    def __init__(self, option):
        self.obser_file = option.obser
        self.reward_file = option.reward
        self.save_state = option.state
        self.save_action = option.action
        self.save_seq = option.seq
        self.save_classify = option.classify
        self.data_init()

    def classified_train_data(self):
        #classified[0]: +1 [1] : -1
        #classified['X'][0]:states
        #classified['X'][1]:actions
        if self.save_classify != None:
            self.classified =  [[[],[]],[[],[]]]
            action0 = np.zeros(10)
            action0[0] = 1
            flag = 0
            for i in range(len(self.sequence)):
                flag = self.sequence[i][-1]['Rt1']
                for j in range(len(self.sequence[i])):
                    if j > 140:
                        ACTION_PERCENT_RETAIN = 0.18
                    else:
                        ACTION_PERCENT_RETAIN = 0.35
                    if ((self.sequence[i][-j-1]['At']==action0).all()) and (random.random() > ACTION_PERCENT_RETAIN):
                        pass
                    else:
                        if flag == int(-100):
                            if j < 20:
                                self.classified[1][0].append(self.sequence[i][-j-1]['St'])
                                self.classified[1][1].append(self.inverse_action(self.sequence[i][-j-1]['At']))
                            else:
                                self.classified[0][0].append(self.sequence[i][-j-1]['St'])
                                self.classified[0][1].append(self.sequence[i][-j-1]['At'])
                        else:
                            self.classified[0][0].append(self.sequence[i][-j-1]['St'])
                            self.classified[0][1].append(self.sequence[i][-j-1]['At'])

    def inverse_action(self, action):
        index = np.where(action == 1)[0]
        new_action = np.zeros(10)
        inds = [0,1,2,4,6,8]
        for i in inds:
            new_action[i]=0.2
        if len(index) != 0 :
            new_action[index[0]] = 0
        return new_action

    def policy_train_data(self):
        if self.save_state != None and self.save_action != None:
            self.states = []
            self.actions = []
            action0 = np.zeros(10)
            action0[0] = 1
            for i in range(len(self.sequence)):
                for j in range(len(self.sequence[i])):
                    if j < 70:
                        ACTION_PERCENT_RETAIN = 0.18
                    else:
                        ACTION_PERCENT_RETAIN = 0.35
                    if ((self.sequence[i][j]['At']==action0).all()) and (random.random() > ACTION_PERCENT_RETAIN):
                        pass
                    else:
                        self.states.append(self.sequence[i][j]['St'])
                        self.actions.append(self.sequence[i][j]['At'])

    def getDataDistribution(self):
        try:
            if(self.actions != None):
                self.act_Distribution = np.zeros(10)
                for i in range(len(self.actions)):
                    self.act_Distribution = self.act_Distribution + self.actions[i]
        except:
            print("error: call parse function first")

    def getClassifyDistribution(self):
        try:
            if(self.classified != None):
                self.act0_Distribution = np.zeros(10)
                self.act1_Distribution = np.zeros(10)
                for i in range(len(self.classified[0][1])):
                    self.act0_Distribution = self.act0_Distribution + self.classified[0][1][i]
                for i in range(len(self.classified[1][1])):
                    self.act1_Distribution = self.act1_Distribution + self.classified[1][1][i]
        except:
            print("error: call parse function first")

    def check_duplicate(self, data):
        if (data[-3]['St'] == data[-2]['St']).all() and (data[-3]['At'] == data[-2]['At']).all() and (data[-3]['Rt1'] == data[-2]['Rt1']):
            data.pop(-3)
            return True
        else:
            return False

    def data_init(self):
        ObserFile = open(self.obser_file,'r')
        trash = ObserFile.readline()
        RewardFile = open(self.reward_file,'r')
        self.sequence = list()
        for r in RewardFile.readlines():
            strings = ObserFile.readline()
            ori_data = self.spliter(strings, "  ")
            data = list()
            counter = 0
            for d in ori_data:
                timeslice = (np.fromstring(d, sep = ",")).astype(int)
                action = np.zeros(10)
                flag = np.zeros(3)
                action[timeslice[0]] = 1
                flag[timeslice[1]] = 1
                if len(timeslice) != 2:
                    state = np.reshape( timeslice[2:], (BOMBR_ROW, BOMBR_COLUMN))
                    state = self.mergeflag(state,flag)
                    info = {'St':state , 'Rt1':REWARD}
                    if counter != 0:
                        data[counter-1]['At'] = action
                        data[counter-1]['St1'] = state
                    if counter > 1 :
                        data[counter-2]['At1'] = action
                    data.append(info)
                else:
                    if counter > 1:
                        data[counter-1]['At'] = action
                        data[counter-2]['At1'] = action
                        data[counter-1]['At1'] = FINALACTION
                        data[counter-1]['St1'] = FINALSTATE
                        data[counter-1]['Rt1'] = float(r.replace("\n",""))
                if len(data) > 2:
                    if(self.check_duplicate(data)):
                        counter = counter-1
                counter = counter+1
            self.sequence.append(data)

    def mergeflag(self, states, flag):
        states = np.delete(states, (0,1,17,18), axis = 0)
        states = np.delete(states, (0,1,17,18), axis = 1)
        states[0][0] = flag[0]
        states[0][1] = flag[1]
        states[0][2] = flag[2]
        return states

    def spliter(self, string, splitElm):
        data = string.split(splitElm)
        data.pop()
        return data

    def save_classify_data(self):
        classify = np.asarray(self.classified)
        if self.save_classify != None:
            np.save(self.save_classify, classify)

    def save_classify_data(self):
        classify = np.asarray(self.classified)
        if self.save_classify != None:
            np.save(self.save_classify, classify)

    def save_npy(self):
        states = np.asarray(self.states)
        actions = np.asarray(self.actions)
        if self.save_state != None and self.save_action != None:
            np.save(self.save_state, states)
            np.save(self.save_action, actions)

    def save_sequence(self):
        sequence = np.asarray(self.sequence)
        if self.save_seq != None:
            np.save(self.save_seq, sequence)
