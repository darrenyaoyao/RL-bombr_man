from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.models import model_from_json
from keras.layers.core import Dense, Dropout, Flatten, Reshape, Merge
import numpy as np
from DQN import DQN
BOMBR_COLUMN = 19
BOMBR_ROW = 19
ACTION_CLASSES = 10

class bombrtrain:
    def __init__(self, option):
        self.option = option
        self.obser_file = option.obser
        self.reward_file = option.reward
        self.model = option.model
        self.weights = option.weights
        self.state_data = option.state
        self.action_data = option.action
        self.classified_data = option.classified
        if self.model == None:
            self.models_init()
        else:
            self.model = model_from_json(open(self.model).read())

    def models_init(self):
        self.model = Sequential()
        self.model.add(Reshape((1, BOMBR_ROW, BOMBR_COLUMN), input_shape=(BOMBR_ROW, BOMBR_COLUMN)))
        self.model.add(Convolution2D(64, 5, 5, activation='relu'))
        self.model.add(Convolution2D(64, 5, 5, activation='relu'))
        self.model.add(Convolution2D(128, 5, 5, activation='relu'))
        self.model.add(Convolution2D(128, 5, 5, activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(ACTION_CLASSES, activation='softmax'))
        open('npyNmodel/model_dpn_Notmerge.json', 'w').write(self.model.to_json())

    def models_policy_train(self):
        if self.state_data != None and self.action_data != None :
            print "start training policy supervised"
            if self.weights != None:
                self.model.load_weights(self.weights)
            else:
                self.weights = 'npyNmodel/model_weight_policy.h5'
            self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            self.model.summary()
            self.states = np.load(self.state_data)
            self.actions = np.load(self.action_data)
            indices = np.arange(len(self.states))
            np.random.shuffle(indices)
            self.states = self.states[indices]
            self.actions = self.actions[indices]
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=8, verbose=0),
                ModelCheckpoint(filepath=self.weights, monitor='val_loss', save_best_only=True, verbose=0)
            ]
            self.model.fit(self.states, self.actions, batch_size=128, nb_epoch=20, verbose=1, validation_split=0.1, callbacks=callbacks)

    def inverse_categorical_crossentropy(self, y_true, y_pred):
        return K.categorical_crossentropy(y_pred, y_true)**(-1)

    def models_inforcement_train(self):
        if self.classified_data != None:
            print "start training policy inforcement"
            if self.weights != None:
                self.model.load_weights(self.weights)
            else:
                self.weights = 'npyNmodel/weight_classified_dpn_Notmerge.h5'
            #fitting for '+1'
            self.classified = np.load(self.classified_data)
            self.states_0 = np.asarray(self.classified[0][0])
            self.actions_0 = np.asarray(self.classified[0][1])
            self.states_1 = np.asarray(self.classified[1][0])
            self.actions_1 = np.asarray(self.classified[1][1])
            #self.states_1 = np.concatenate((states_0 , states_3),axis=0)
            #print len(self.states_1)
            #self.actions_1 = np.concatenate ((actions_0 , actions_3),axis = 0)
            #print len(self.actions_1)
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=8, verbose=0),
                ModelCheckpoint(filepath=self.weights, monitor='val_loss', save_best_only=True, verbose=0)
            ]
            indices = np.arange(len(self.states_0))
            np.random.shuffle(indices)
            self.states_0 = self.states_0[indices]
            self.actions_0 = self.actions_0[indices]
            ind = np.arange(len(self.states_1))
            np.random.shuffle(ind)
            self.states_1 = self.states_1[ind]
            self.actions_1 = self.actions_1[ind]
            
            self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            self.model.summary()
            self.model.fit(self.states_0, self.actions_0, batch_size=128, nb_epoch=20, verbose=1, validation_split=0.1, callbacks=callbacks)
            '''          
            self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
            self.model.summary()
            self.model.fit(self.states_1, self.actions_1, batch_size=64, nb_epoch=20, verbose=1, validation_split=0.1, callbacks=callbacks)
            '''

    def test_predict(self):
        self.model.load_weights(self.weights)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        for x in self.states:
            state = np.zeros((1, BOMBR_ROW, BOMBR_COLUMN))
            state[0] = x
            action = self.model.predict_classes(state)
            print (action)

    def dqnmodel_init(self, load_weights=False, weights_file="../dqnmodel_weight.h5"):
        self.seq = np.load(self.option.seq)
        self.dqn_datainit()
        state_model = Sequential()
        state_model.add(Reshape((1, BOMBR_ROW, BOMBR_COLUMN), input_shape=(BOMBR_ROW, BOMBR_COLUMN)))
        state_model.add(Convolution2D(64, 3, 3, activation='relu'))
        state_model.add(Convolution2D(64, 3, 3, activation='relu'))
        state_model.add(Dropout(0.25))
        state_model.add(Flatten())
        state_model.add(Dense(128, activation='relu'))

        action_model = Sequential()
        action_model.add(Dense(32, input_shape=(10,), activation='relu'))
        action_model.add(Dense(32, activation='relu'))
        
        merged = Merge([state_model, action_model], mode='concat')
        final_model = Sequential()
        final_model.add(merged)
        final_model.add(Dense(200, activation='relu'))
        final_model.add(Dense(200, activation='relu'))
        final_model.add(Dense(1, activation='linear'))
        open('dqnmodel.json', 'w').write(final_model.to_json())
        self.dqnmodel = DQN(final_model, load_weights, weights_file)

    def dqn_datainit(self):
      self.all_action = list()
      for i in range(10):
        action = np.zeros(10)
        action[i] = 1
        self.all_action.append(action)
      self.dqn_data = list()
      for i in range(len(self.seq)):
        for j in range(len(self.seq[i])):
          if (len(self.seq[i])-j) < 5 :
            self.dqn_data.append(self.seq[i][j])

    def dqn_train(self):
      self.dqnmodel.train(self.dqn_data, self.all_action, 0.95)

    def dqn_train_test(self):
      for game in self.seq:
        print "Game final reward: " + str(game[-1]['Rt1'])
        for i in range(len(game)):
          if (len(game)-i) < 5 :
            x = np.zeros((1, BOMBR_ROW, BOMBR_COLUMN))
            x[0] = game[i]['St']
            a = np.zeros((1, 10))
            a[0] = game[i]['At']
            Q = self.dqnmodel.predict([x, a])
            if game[-1]['Rt1'] == -1:
              print Q[0][0]
