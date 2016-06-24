from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.layers.convolutional import Convolution2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
BOMBR_COLUMN = 19
BOMBR_ROW = 19
FINALSTATE = np.full((19,19),3.0)
REWARD = 0
ACTION_CLASSES = 10

class bombrtrain:
   def __init__(self, option):
      self.obser_file = option.obser
      self.reward_file = option.reward
      self.models_init()
      self.data_init()
      self.parse_policy_train_data()

   def parse_policy_train_data(self):
      self.states = []
      self.actions = []
      state_action_pairs = []
      for i in range(len(self.sequence)):
         for j in range(len(self.sequence[i])):
            if(self.check_duplicate(state_action_pairs, self.sequence[i][j])):
               self.states.append(self.sequence[i][j]['St'])
               self.actions.append(self.sequence[i][j]['At'])

   def check_duplicate(self, state_action_pairs, data):
      state_action_pair = (data['St'], data['At'])
      if state_action_pair in state_action_pairs:
         return False
      else:
         state_action_pairs.append(state_action_pair)
         return True

   def models_init(self):
      #Todo
      self.model = Sequential()
      self.model.add(Reshape((1, BOMBR_ROW, BOMBR_COLUMN), input_shape=(BOMBR_ROW, BOMBR_COLUMN)))
      self.model.add(Convolution2D(64, 3, 3, activation='relu'))
      self.model.add(Convolution2D(64, 3, 3, activation='relu'))
      self.model.add(Dropout(0.25))
      self.model.add(Flatten())
      self.model.add(Dense(128, activation='relu'))
      self.model.add(Dropout(0.5))
      self.model.add(Dense(ACTION_CLASSES, activation='softmax'))
      open('model.json', 'w').write(self.model.to_json())

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
         for counter, d in enumerate(ori_data):
             timeslice = np.fromstring(d, sep = ",")
             timeslice = timeslice.astype(np.int32)
             action=np.zeros(10)
             action[timeslice[0]] = 1
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

   def models_policy_train(self):
      self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
      self.model.summary()
      callbacks = [
          EarlyStopping(monitor='val_loss', patience=5, verbose=0),
          ModelCheckpoint(filepath="model_weight.h5", monitor='val_loss', save_best_only=True, verbose=0)
      ]
      self.model.fit(np.asarray(self.states), np.asarray(self.actions), batch_size=128, nb_epoch=20, verbose=1, validation_split=0.1, callbacks=callbacks)

