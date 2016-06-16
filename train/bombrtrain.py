from keras.models import Sequential
import numpy
BOMBR_COLUMN = 19
BOMBR_ROW = 19

class bombrtrain:
   def __init__(self, obser_file, reward_file, option):
      self.obser_file = obser_file
      self.reward_file = reward_file
      #We have two model: one that we call train_model is for training , 
      #                   the other that we call value_model is evaluating the freezing weights
      self.models_init()

   def models_init(self):
      self.train_model = build_model()
      self.value_model = build_model()

   def build_model(self):
      state_model = self.build_state_model()
      action_model = self.build_action_model()

      return self.build_final_model(state_model, action_model)

   def build_state_model(self):
      state_model = Sequential()
      state_model.add(Convolution2D(64, 3, 3, border_mode='same', 
         input_shape=(1, BOMBR_COLUMN, BOMBR_ROW)), activation='relu')
      state_model.add(Convolution2D(64, 3, 3, activation='relu')
      state_model.add(MaxPooling2D(pool_size=(2, 2)))
      state_model.add(Dropout(0.25))
      state_model.add(Flatten())
      state_model.add(Dense(128, activation='relu'))

      return state_model

   def build_action_model(self):
      action_model = Sequential()
      action_model.add(Dense(32, input_dim=10))

      return action_model

   def build_final_model(self, state_model, action_model):
      merge = Merge([state_model, action_model], mode="concat")
      final_model = Sequential()
      final_model.add(merge)      
      final_model.add(Dense(128, activation='relu'))
      final_model.add(Dense(1))

      return final_model
   
   def value_based_training(self):
      # DQN Todo
      #self.train_model.compile()

   def dqn_mse_target(self):
      #DQN Todo
      


