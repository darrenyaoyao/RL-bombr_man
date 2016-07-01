from keras.models import Sequential
<<<<<<< HEAD
from keras.layers.convolutional import Convolution2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.models import model_from_json
from keras.layers.core import Dense, Dropout, Flatten, Reshape, Merge
=======
from keras.layers.core import Dense, Dropout, Flatten, Reshape, Merge
from keras.layers.convolutional import Convolution2D, ZeroPadding2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import model_from_json
>>>>>>> master
import numpy as np
<<<<<<< HEAD
#from DQN import DQN
=======
from DQN import DQN
<<<<<<< HEAD
=======
>>>>>>> d92eb4deeb01cea0b68c9a235f81a36e4c0b66d4
>>>>>>> master
BOMBR_COLUMN = 15
BOMBR_ROW = 15
ACTION_CLASSES = 10
FEATURE_CLASSES = 6

class bombrtrain:
    def __init__(self, option):
        self.option = option
        self.obser_file = option.obser
        self.reward_file = option.reward
        self.model = option.model
        self.weights = option.weights
        self.classified_data = option.classify
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
<<<<<<< HEAD
        self.model.add(Convolution2D(128, 5, 5, activation='relu'))
        self.model.add(Convolution2D(64, 3, 3, activation='relu'))
=======
        self.model.add(ZeroPadding2D(padding=(1, 1)))
        self.model.add(Convolution2D(128, 3, 3, activation='relu'))
        self.model.add(ZeroPadding2D(padding=(1, 1)))
        self.model.add(Convolution2D(128, 3, 3, activation='relu'))
>>>>>>> master
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(ACTION_CLASSES, activation='softmax'))
<<<<<<< HEAD
        open('modelNweight/model_withflag_all.json', 'w').write(self.model.to_json())
=======
<<<<<<< HEAD
        open('./model_default.json', 'w').write(self.model.to_json())
=======
        open('../model/new_model.json', 'w').write(self.model.to_json())
>>>>>>> d92eb4deeb01cea0b68c9a235f81a36e4c0b66d4
>>>>>>> master

    def models_inforcement_train(self):
        if self.classified_data != None:
            print "start training policy inforcement"
        if self.weights != None:
            self.model.load_weights(self.weights)
        else:
            self.weights = './weight_inforcement_default.h5'                                
        #fitting for '+1'
        self.classified = np.load(self.classified_data)
        self.states_0 = np.asarray(self.classified[0][0])
        self.actions_0 = np.asarray(self.classified[0][1])
        self.states_1 = np.asarray(self.classified[1][0])
        self.actions_1 = np.asarray(self.classified[1][1])
        #self.states_2 = np.concatenate((states_0 , states_1),axis=0)
        #self.actions_2 = np.concatenate ((actions_0 , actions_1),axis=0)
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
        self.model.fit(self.states_0, self.actions_0, batch_size=128, nb_epoch=30, verbose=1, validation    _split=0.1, callbacks=callbacks)
        '''
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        self.model.summary()
        self.model.fit(self.states_1, self.actions_1, batch_size=32, nb_epoch=20, verbose=1, validation_    split=0.1, callbacks=callbacks)
        '''
        
    def models_policy_train(self):
<<<<<<< HEAD
        if self.state_data != None and self.action_data != None :
            print "start training policy supervised"
            if self.weights != None:
                self.model.load_weights(self.weights)
            else:
                self.weights = 'modelNweight/weight_withflag_all_policy.h5'
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
=======
        if self.weights != None:
            self.model.load_weights(self.weights)
        else:
<<<<<<< HEAD
            self.weights = './weight_default.h5'
=======
            self.weights = '../model/new_model_weight.h5'
>>>>>>> d92eb4deeb01cea0b68c9a235f81a36e4c0b66d4
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()
        self.states = np.load(self.state_data)
        self.actions = np.load(self.action_data)
        indices = np.arange(len(self.states))
        np.random.shuffle(indices)
        self.states = self.states[indices]
        self.actions = self.actions[indices]
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, verbose=0),
            ModelCheckpoint(filepath=self.weights, monitor='val_loss', save_best_only=True, verbose=0)
        ]
        self.model.fit(self.states, self.actions, batch_size=128, nb_epoch=50, verbose=1, validation_split=0.1, callbacks=callbacks)
>>>>>>> master

    def test_predict(self):
        self.model.load_weights(self.weights)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        for x in self.states:
            state = np.zeros((1, BOMBR_ROW, BOMBR_COLUMN))
            state[0] = x
            action = self.model.predict_classes(state)
            print (action)

<<<<<<< HEAD
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
=======
    def dqnmodel_init(self, load_weights=False, weights_file="./model/dqn_new_reward.h5"):
      self.seq = np.load(self.option.seq)
      state_model = Sequential()
      state_model.add(Reshape((1, BOMBR_ROW, BOMBR_COLUMN), input_shape=(BOMBR_ROW, BOMBR_COLUMN)))
      state_model.add(Convolution2D(64, 5, 5, activation='relu'))
      state_model.add(Convolution2D(64, 3, 3, activation='relu'))
      state_model.add(Dropout(0.25))
      state_model.add(Flatten())
      state_model.add(Dense(256, activation='relu'))

      action_model = Sequential()
      action_model.add(Dense(32, input_shape=(10,), activation='relu'))
      action_model.add(Dense(64, activation='relu'))

      merged = Merge([state_model, action_model], mode='concat')
      final_model = Sequential()
      final_model.add(merged)
      final_model.add(Dense(256, activation='relu'))
      final_model.add(Dense(256, activation='relu'))
      final_model.add(Dense(1, activation='linear'))
      open('dqnmodel.json', 'w').write(final_model.to_json())
      self.dqnmodel = DQN(final_model, load_weights, weights_file)
>>>>>>> master

    def dqn_train(self):
      self.dqnmodel.train(self.seq, self.all_action, 0.96)

    def sarsa_train(self):
      self.dqnmodel.sarsa_train(self.seq, 0.96)

    def dqn_train_test(self):
      for game in self.seq:
        print "Game final reward: " + str(game[-1]['Rt1'])
        for i in range(len(game)):
          if (len(game)-i) < 15 :
            x = np.zeros((1, BOMBR_ROW, BOMBR_COLUMN))
            x[0] = game[i]['St']
            a = np.zeros((1, 10))
            a[0] = game[i]['At']
            Q = self.dqnmodel.predict([x, a])
            print Q[0][0]
    
    def model_feature_init(self):
        states_model = Sequential()
        states_model.add(Reshape((1, BOMBR_ROW, BOMBR_COLUMN), input_shape=(BOMBR_ROW, BOMBR_COLUMN)))
        states_model.add(Convolution2D(64, 3, 3, activation='relu'))
        states_model.add(Convolution2D(64, 3, 3, activation='relu'))
        states_model.add(Dropout(0.25))
        states_model.add(Flatten())
        states_model.add(Dense(256, activation='relu'))

        features_model = Sequential()
        features_model.add(Dense(32, input_shape=(6,), activation='relu'))
        features_model.add(Dense(64, activation='relu'))

        merged = Merge([states_model, features_model], mode='concat')
        self.final_model = Sequential()
        self.final_model.add(merged)
        self.final_model.add(Dense(512, activation='relu'))
        self.final_model.add(Dense(512, activation='relu'))
        self.final_model.add(Dense(ACTION_CLASSES, activation='softmax'))
        open('model_feature.json', 'w').write(self.final_model.to_json())

    def model_feature_train(self):
        if self.weights != None:
            self.final_model.load_weights(self.weights)
        else:
            self.weights = 'model_feature_weight.h5'
        self.final_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.final_model.summary()
        self.states = np.load(self.state_data)
        self.actions = np.load(self.action_data)
        self.features = np.load(self.feature_data)
        indices = np.arange(len(self.states))
        np.random.shuffle(indices)
        self.states = self.states[indices]
        self.actions = self.actions[indices]
        self.features = self.features[indices]
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, verbose=0),
            ModelCheckpoint(filepath=self.weights, monitor='val_loss', save_best_only=True, verbose=0)
        ]
        self.final_model.fit([self.states, self.features], self.actions, batch_size=128, nb_epoch=50, verbose=1, validation_split=0.1, callbacks=callbacks)

