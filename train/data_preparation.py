import argparse
import theano
import numpy as np
from parseData import parseData

parser = argparse.ArgumentParser(description='Bombr Game')

parser.add_argument('-O', action='store', dest='obser')
parser.add_argument('-R', action='store', dest='reward')
parser.add_argument('-S', action='store', dest='state')
parser.add_argument('-A', action='store', dest='action')
parser.add_argument('-Q', action='store', dest='seq')
parser.add_argument('-C', action='store', dest='classify')

Bombr_data = parseData(parser.parse_args())
Bombr_data.policy_train_data()
Bombr_data.getDataDistribution()
for i in range(len(Bombr_data.sequence[0])):
    print "data[ "+str(i)+" ] : "
    print "At: "+str(Bombr_data.sequence[0][i]['At'])
    print "At1: "+str(Bombr_data.sequence[0][i]['At1'])
Bombr_data.save_npy()
