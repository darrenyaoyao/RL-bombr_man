from keras.models import model_from_json
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
FINALSTATE = np.full((19,19),3.0)
BOMBR_COLUMN = 19
BOMBR_ROW = 19

class DQN:
   def __init__(self, model):
      self.model = model
      json_string = model.to_json()
      self.evalute_model = model_from_json(json_string)

   def train(self, data, actions, gamma, batch_size=32, nb_epoch=10, 
         nb_iter=10, optimizer=SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)):
      self.actions = actions
      self.optimizer = optimizer
      self.gamma = gamma
      states = []
      rewards = []
      actions = []
      next_states = []
      for d in data:
         states.append(d['St'])
         rewards.append(d['Rt1'])
         actions.append(d['At'])
         next_states.append(d['St1'])

      self.model.compile(loss='mean_squared_error', optimizer=optimizer)
      self.model.summary()
      callbacks = [
         EarlyStopping(monitor='val_loss', patience=5, verbose=0),
         ModelCheckpoint(filepath="dqlmodel_weight.h5", monitor='val_loss', save_best_only=True, verbose=0)
      ]
      self.save_model_weight()
      for x in range(nb_epoch):
         self.update_evalute_model_weight()
         self.update_target(rewards, next_states)
         self.model.fit([states, actions], self.targets, batch_size, nb_iter)
         self.save_model_weight()

   def update_target(self, reward, next_state):
      self.targets = []
      for i in range(len(reward)):
         self.targets.append(reward[i]+self.gamma*self.get_maxQ(next_state[i]))

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

   def get_maxQ(self, state):
      maxQ = float('-inf')
      x = np.zeros((1, BOMBR_ROW, BOMBR_COLUMN))
      x[0] = state
      for action in self.actions:
         maxQ = max(maxQ, self.evalute_model.predict([x, action]))
      return maxQ