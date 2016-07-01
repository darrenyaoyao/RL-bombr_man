import argparse
import theano
import numpy as np
from parseData import parseData

parser = argparse.ArgumentParser(description='Bombr Game')

parser.add_argument('-O', action='store', dest='obser')
parser.add_argument('-R', action='store', dest='reward')
parser.add_argument('-S', action='store', dest='state')
parser.add_argument('-A', action='store', dest='action')
parser.add_argument('-C', action='store', dest='classify')
parser.add_argument('-Q', action='store', dest='seq')
parser.add_argument('-C', action='store', dest='classify')

Bombr_data = parseData(parser.parse_args())
Bombr_data.policy_train_data()
Bombr_data.classified_train_data():
Bombr_data.getDataDistribution()
Bombr_data.getClassifyDistribution()
Bombr_data.save_npy()
Bombr_data.save_sequence()
Bombr_data.save_classify_data()
#print ("action distribution: ")
#print (Bombr_data.act_Distribution)
#print ("After classifiy, action distribution: ")
#print ("Reward +1 & -1 front")
#print (Bombr_data.act0_Distribution)
#print ("Reward -1 last 70 steps ")
#print (Bombr_data.act1_Distribution)
