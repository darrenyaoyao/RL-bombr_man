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
      for i in range(len(self.sequence)):
          for j in range(len(self.sequence[i])):
              self.states.append(self.sequence[i][j][St])
              self.actions.append(self.sequence[i][j][At])
   def models_init(self):
      #Todo
      model = Sequential()
      model.add(Reshape((1, bombr_rows, bombr_cols), input_shape=(bombr_rows, bombr_cols)))
      model.add(Convolution2D(64, 3, 3, activation='relu'))
      model.add(Convolution2D(64, 3, 3, activation='relu'))
      model.add(Dropout(0.25))
      model.add(Flatten())
      model.add(Dense(128, activation='relu'))
      model.add(Dropout(0.5))
      model.add(Dense(action_classes, activation='softmax'))
      open('model.json', 'w').write(model.to_json())
      model.save_weights('weights.h5', overwrite=True)

   def models_policy_train(self):
      model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
      model.summary()
      model.fit(self.states, self.actions, batch_size=128, nb_epoch=20, verbose=1, validation_split=0.1)
   def data_init(self):
      #Todo
