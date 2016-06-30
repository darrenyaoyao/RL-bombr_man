from keras.models import model_from_json
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
FINALSTATE = np.full((19,19),3.0)
BOMBR_COLUMN = 19
BOMBR_ROW = 19

class DQN:
   def __init__(self, model, load_weight=False, weight_file="./model/dqn_new_reward.h5"):
      self.model = model
      json_string = model.to_json()
      self.weight_file = weight_file
      print self.weight_file
      self.evalute_model = model_from_json(json_string)
      if load_weight:
         self.model.load_weights(weight_file)
         self.evalute_model.load_weights(weight_file)

   def train(self, data, actions, gamma=0.999, batch_size=32, nb_epoch=200,
         nb_iter=20, optimizer='adam'):
      self.actions = actions
      self.optimizer = optimizer
      self.gamma = gamma
      states = [] #list of numpy array
      rewards = []
      actions = []
      next_states = []
      '''for d in data:
         states.append(d['St'])
         rewards.append(d['Rt1'])
         actions.append(d['At'])
         next_states.append(d['St1'])'''
      num = 1
      for i in range(len(data)):
         if len(data[i])-num >= 0:
            for j in range(len(data[i])-num, len(data[i])):
               states.append(data[i][j]['St'])
               rewards.append(data[i][j]['Rt1'])
               actions.append(data[i][j]['At'])
               next_states.append(data[i][j]['St1'])
         else:
            for j in range(0, len(data[i])):
               states.append(data[i][j]['St'])
               rewards.append(data[i][j]['Rt1'])
               actions.append(data[i][j]['At'])
               next_states.append(data[i][j]['St1'])
      npstates = np.array(states)
      npactions = np.array(actions)

      self.model.compile(loss='mean_squared_error', optimizer=optimizer)
      self.model.summary()
      self.save_model_weight()
      for x in range(nb_epoch):
         print "All Date Epoch "+str(x)+"/"+str(nb_epoch)
         self.update_evalute_model_weight()
         self.update_target(rewards, next_states)
         print "Shuffle data"
         #self.shuffle(npstates, npactions)
         print "Start fit"
         print self.targets
         self.model.fit([npstates, npactions], self.targets, batch_size, nb_iter, verbose=1, validation_split=0, shuffle=True)
         print "Finish fit"
         self.save_model_weight()
         #update_data
         num += 1
         for i in range(len(data)):
            if len(data[i])-num >= 0:
               states.append(data[i][len(data[i])-num]['St'])
               actions.append(data[i][len(data[i])-num]['At'])
               rewards.append(data[i][len(data[i])-num]['Rt1'])
               next_states.append(data[i][len(data[i])-num]['St1'])
         npstates = np.array(states)
         npactions = np.array(actions)
         self.model.save_weights(self.weight_file)

   def sarsa_train(self, data, gamma=0.999, batch_size=32, nb_epoch=200,
         nb_iter=20, optimizer='adam'):
      self.optimizer = optimizer
      self.gamma = gamma
      states = [] #list of numpy array
      rewards = []
      actions = []
      next_states = []
      num = 1
      for i in range(len(data)):
         if len(data[i])-num >= 0:
            for j in range(len(data[i])-num, len(data[i])):
               states.append(data[i][j]['St'])
               rewards.append(data[i][j]['Rt1'])
               actions.append(data[i][j]['At'])
               next_states.append(data[i][j]['St1'])
         else:
            for j in range(0, len(data[i])):
               states.append(data[i][j]['St'])
               rewards.append(data[i][j]['Rt1'])
               actions.append(data[i][j]['At'])
               next_actions.append(data[i][j]['At1'])
               next_states.append(data[i][j]['St1'])
      npstates = np.array(states)
      npactions = np.array(actions)
      npnext_actions = np.array(next_actions)
      npnext_states = np.array(next_states)

      self.model.compile(loss='mean_squared_error', optimizer=optimizer)
      self.model.summary()
      self.save_model_weight()
      for x in range(nb_epoch):
         print "All Date Epoch "+str(x)+"/"+str(nb_epoch)
         self.update_evalute_model_weight()
         self.targets = self.evalute_model.predict([npnext_states, npnext_actions])
         print "Start fit"
         print self.targets
         self.model.fit([npstates, npactions], self.targets, batch_size, nb_iter, verbose=1, validation_split=0, shuffle=True)
         print "Finish fit"
         self.save_model_weight()
         #update_data
         num += 1
         for i in range(len(data)):
            if len(data[i])-num >= 0:
               states.append(data[i][len(data[i])-num]['St'])
               actions.append(data[i][len(data[i])-num]['At'])
               rewards.append(data[i][len(data[i])-num]['Rt1'])
               next_actions.append(data[i][len(data[i])-num]['At1'])
               next_states.append(data[i][len(data[i])-num]['St1'])
         npstates = np.array(states)
         npactions = np.array(actions)
         npnext_actions = np.array(next_actions)
         npnext_states = np.array(next_states)
         self.model.save_weights(self.weight_file)

   def update_target(self, reward, next_state):
      self.targets = []
      x, a = self.collect_predict_data(reward, next_state)
      print "Predict"
      Q = self.evalute_model.predict([x, a])
      print "Get max Q"
      for i in range(len(reward)):
         if (next_state[i] == FINALSTATE).all():
            self.targets.append(reward[i])
            print reward[i]
            print "================"
         else:
            a = self.get_maxQ(Q[10*i:10*(i+1)])
            self.targets.append(reward[i]+self.gamma*a)
            print reward[i]+self.gamma*a
      self.targets = np.array(self.targets)

   def save_model_weight(self):
      self.model_weights = []
      for layer in self.model.layers:
         self.model_weights.append(layer.get_weights())

   def update_evalute_model_weight(self):
      i = 0
      for layer in self.evalute_model.layers:
         layer.set_weights(self.model_weights[i])
         i += 1
      self.evalute_model.compile(loss='mean_squared_error', optimizer=self.optimizer)

   def get_maxQ(self, Q):
      maxQ = float('-inf')
      for i in range(10):
         maxQ = max(maxQ, Q[i][0])
      return maxQ

   def shuffle(self, npstates, npactions):
      indices = np.arange(len(self.targets))
      np.random.shuffle(indices)
      npstates = npstates[indices]
      npacions = npactions[indices]
      self.targets = self.targets[indices]

   def collect_predict_data(self, reward, next_state):
      x = np.zeros((10*len(reward), BOMBR_ROW, BOMBR_COLUMN))
      a = np.zeros((10*len(reward), 10))
      for i in range(len(reward)):
        for j in range(10):
            x[i*10+j] = next_state[i]
            a[i*10+j] = self.actions[j]
      return x ,a

   def predict(self, data):
      return self.evalute_model.predict(data)
