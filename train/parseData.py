import numpy as np
import random
BOMBR_COLUMN = 19
BOMBR_ROW = 19
FINALSTATE = np.full((19,19),3.0)
#FINALSTATE = np.full(361,3.0)
REWARD = 0
ACTION_PERCENT_RETAIN = 0.2

class parseData:
    def __init__(self, option):
        self.obser_file = option.obser
        self.reward_file = option.reward
        self.save_state = option.state
        self.save_action = option.action
        self.data_init()

    def policy_train_data(self):
        self.states = []
        self.actions = []
        action0 = np.zeros(10)
        action0[0] = 1
        for i in range(len(self.sequence)):
            for j in range(len(self.sequence[i])):
                if ((self.sequence[i][j]['At']==action0).all()) and (random.random() > ACTION_PERCENT_RETAIN):
                    pass
                else:
                    self.states.append(self.sequence[i][j]['St'])
                    self.actions.append(self.sequence[i][j]['At'])
        print len(self.states)

    def getDataDistribution(self):
        try:
            if(self.actions != None):
                self.act_Distribution = np.zeros(10)
                for i in range(len(self.actions)):
                    self.act_Distribution = self.act_Distribution + self.actions[i]
        except:
            print("call parse function first")

    def check_duplicate(self, data):
        if (data[-1]['St'] == data[-2]['St']).all() and (data[-1]['At'] == data[-2]['At']).all() and (data[-1]['Rt1'] == data[-2]['Rt1']):
            data.pop(-2)
            return True
        else:
            return False

# sequence for whole dataset
    def data_init(self):
        ObserFile = open(self.obser_file,'r')
        trash = ObserFile.readline()
        RewardFile = open(self.reward_file,'r')
        #every game sequence's data
        self.sequence = list()
        for r in RewardFile.readlines():
            strings = ObserFile.readline()
            ori_data = self.spliter(strings, "  ")
            #every time step's data
            data = list()
            counter = 0
            for d in ori_data:
                timeslice = np.fromstring(d, sep = ",")
                timeslice = timeslice.astype(int)
                action=np.zeros(10)
                action[timeslice[0]] = 1
                if len(timeslice) != 1:
                    state = np.reshape( timeslice[1:], (BOMBR_ROW, BOMBR_COLUMN))
                    #state = timeslice[1:]
                    info = {'St':state , 'Rt1':REWARD}
                    if counter != 0:
                        data[counter-1]['At'] = action
                        data[counter-1]['St1'] = state
                else:
                    data[counter-1]['At'] = action
                    data[counter-1]['St1'] = FINALSTATE
                    data[counter-1]['Rt1'] = r
                if len(data) > 1:
                    if(self.check_duplicate(data)):
                        counter = counter-1
                data.append(info)
                counter = counter+1
            self.sequence.append(data)

    def spliter(self, string, splitElm):
        data = string.split(splitElm)
        data.pop()
        return data

    def save_npy(self):
        states = np.asarray(self.states)
        actions = np.asarray(self.actions)
        np.save(self.save_state, states)
        np.save(self.save_action, actions)
