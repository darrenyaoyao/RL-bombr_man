from bombrtrain import bombrtrain
import argparse
import theano
LOAD_WEIGHTS = True

parser = argparse.ArgumentParser(description='Bombr Game')

parser.add_argument('-O', action="store", dest='obser')
parser.add_argument('-R', action="store", dest='reward')
parser.add_argument('-M', action="store", dest='model')
parser.add_argument('-W', action="store", dest='weights')
parser.add_argument('-S', action="store", dest='state')
parser.add_argument('-A', action="store", dest='action')
parser.add_argument('-Q', action="store", dest='seq')

Bombr_train = bombrtrain(parser.parse_args())
Bombr_train.dqnmodel_init(LOAD_WEIGHTS, "dqnmodel_weight.h5")
#Bombr_train.dqnmodel_init()
#Bombr_train.dqn_train_test()
Bombr_train.dqn_train()
