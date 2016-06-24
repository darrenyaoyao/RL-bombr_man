from bombrtrain import bombrtrain
import argparse
import theano
#theano.config.device = 'gpu'
#theano.config.floatX = 'float32'

parser = argparse.ArgumentParser(description='Bombr Game')

parser.add_argument('-O', action="store", dest='obser')
parser.add_argument('-W', action="store", dest='reward')

Bombr_train = bombrtrain(parser.parse_args())
for i in range(len(Bombr_train.sequence[0])):
    print Bombr_train.sequence[0][i]['At']
#Bombr_train.models_policy_train()
#Bombr_train.models_save_weight()
