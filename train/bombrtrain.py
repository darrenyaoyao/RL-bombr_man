from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.layers.convolutional import Convolution2D
import numpy
bombr_cols = 19
bombr_rows = 19
action_classes = 10
class bombrtrain:
   def __init__(self, obser_file, reward_file, option):
      self.obser_file = obser_file
      self.reward_file = reward_file
      self.models_init()

   def parse_policy_train_data(self):
      for i in range(len(sequence)):
          for j in range(len(sequence[i])):
              self.states.append(sequence[i][j][St])
              self.actions.append(sequence[i][j][At])

   def models_init(self):
      #Todo
      self.model = Sequential()
      self.model.add(Reshape((1, bombr_rows, bombr_cols), input_shape=(bombr_rows, bombr_cols)))
      self.model.add(Convolution2D(64, 3, 3, activation='relu'))
      self.model.add(Convolution2D(64, 3, 3, activation='relu'))
      self.model.add(Dropout(0.25))
      self.model.add(Flatten())
      self.model.add(Dense(128, activation='relu'))
      self.model.add(Dropout(0.5))
      self.model.add(Dense(action_classes, activation='softmax'))
      open('model.json', 'w').write(model.to_json())

   def models_policy_train(self):
      self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
      self.model.summary()
      self.model.fit(self.states, self.actions, batch_size=128, nb_epoch=20, verbose=1, validation_split=0.1)

   def models_save_weight(self):
      self.model.save_weight('model_weight.h5')