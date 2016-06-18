from keras.models import Sequential
import numpy as np
BOMBR_COLUMN = 19
BOMBR_ROW = 19
FINALSTATE = np.full((19,19),3)
REWARD = 0

class bombrtrain:
    def __init__(self, obser_file, reward_file, option):
        self.obser_file = obser_file
        self.reward_file = reward_file
        self.data_init()
        #We have two model: one that we call train_model is for training , 
        #                   the other that we call value_model is evaluating the freezing weights
     #   self.models_init()

#    def models_init(self):
        #Todo

    def data_init(self):
       # ObserFN = 'default.obser'
       # RewardFN = 'default.reward'
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
            for counter, d in enumerate(ori_data):
                timeslice = np.fromstring(d, sep = ",")
                action = timeslice[0]
                if len(timeslice) != 1:
                    state = np.reshape( timeslice[1:], (BOMBR_ROW, BOMBR_COLUMN))
                    info = {'St':state , 'Rt1':REWARD}
                    if counter != 0:
                        data[counter-1]['At']=action
                        data[counter-1]['St1']=state
                else:
                    data[counter-1]['At']=action
                    data[counter-1]['St1']=FINALSTATE
                    data[counter-1]['Rt1']=r
                data.append(info)
            self.sequence.append(data)

    def spliter(self, string, splitElm):
        data = string.split(splitElm)
        data.pop()
        return data




