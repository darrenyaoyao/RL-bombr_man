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
      #Todo

   def data_init(self):
      #Todo
      


