import numpy as np
import random
BOMBR_COLUMN = 19
BOMBR_ROW = 19
FINALSTATE = np.full((19,19),3.0)
#FINALSTATE = np.full(361,3.0)
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
        self.classified =  [[[],[]],[[],[]]]
        action0 = np.zeros(10)
        action0[0] = 1
        flag = 0
        for i in range(len(self.sequence)):
            flag = self.sequence[i][-1]['Rt1']
            for j in range(len(self.sequence[i])):
                if j > 140:
                    ACTION_PERCENT_RETAIN = 0.15
                else:
                    ACTION_PERCENT_RETAIN = 0.30
                if ((self.sequence[i][-j-1]['At']==action0).all()) and (random.random() > ACTION_PERCENT_RETAIN):
                    pass
                else:
                    if flag == int(-1):
                        if j < 70:
                            self.classified[1][0].append(self.sequence[i][-j-1]['St'])
                            self.classified[1][1].append(self.sequence[i][-j-1]['At'])
                        else:
                            self.classified[0][0].append(self.sequence[i][-j-1]['St'])
                            self.classified[0][1].append(self.sequence[i][-j-1]['At'])
                    else:
                        self.classified[0][0].append(self.sequence[i][-j-1]['St'])
                        self.classified[0][1].append(self.sequence[i][-j-1]['At'])


    def policy_train_data(self):
        self.states = []
        self.actions = []
        action0 = np.zeros(10)
        action0[0] = 1
        for i in range(len(self.sequence)):
            for j in range(len(self.sequence[i])):
                if j < 70:
                    ACTION_PERCENT_RETAIN = 0.15
                else:
                    ACTION_PERCENT_RETAIN = 0.30
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

# sequence for whole dataset
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
                action[timeslice[0]] = 1
                if len(timeslice) != 1:
                    state = np.reshape( timeslice[1:], (BOMBR_ROW, BOMBR_COLUMN))
                    info = {'St':state , 'Rt1':REWARD}
                    if counter != 0:
                        data[counter-1]['At'] = action
                        data[counter-1]['St1'] = state
                    data.append(info)
                else:
                    data[counter-1]['At'] = action
                    data[counter-1]['St1'] = FINALSTATE
                    data[counter-1]['Rt1'] = int(r.replace("\n",""))
                if len(data) > 2:
                    if(self.check_duplicate(data)):
                        counter = counter-1
                counter = counter+1
            self.sequence.append(data)

    def spliter(self, string, splitElm):
        data = string.split(splitElm)
        data.pop()
        return data
    
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
