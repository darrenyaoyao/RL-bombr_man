from bombrtrain import bombrtrain
from loadData import loadData
import argparse
import theano
LOAD_WEIGHTS = True

parser = argparse.ArgumentParser(description='Bombr Game')

parser.add_argument('-O', action="store", dest='obser')
parser.add_argument('-W', action="store", dest='reward')

Bombr_data.policy_train_data()
Bombr_data.getDataDistribution()
print(Bombr_data.act_Distribution)
#Bombr_train.models_policy_train(LOAD_WEIGHTS)
#Bombr_train.test_predict()
