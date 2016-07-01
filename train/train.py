import argparse
import theano
from bombrtrain import bombrtrain

parser = argparse.ArgumentParser(description='Bombr Game')

parser.add_argument('-O', action="store", dest='obser')
parser.add_argument('-R', action="store", dest='reward')
parser.add_argument('-M', action="store", dest='model')
parser.add_argument('-W', action="store", dest='weights')
parser.add_argument('-S', action="store", dest='state')
parser.add_argument('-A', action="store", dest='action')
<<<<<<< HEAD
parser.add_argument('-C', action="store", dest='classified')

Bombr_train = bombrtrain(parser.parse_args())
Bombr_train.models_policy_train()
#Bombr_train.models_inforcement_train()
=======
parser.add_argument('-C', action="store", dest='classify')
parser.add_argument('-F', action="store", dest='feature')

Bombr_train = bombrtrain(parser.parse_args())
Bombr_train.models_policy_train()
<<<<<<< HEAD
Bombr_train.models_inforcement_train()
#Bombr_train.model_feature_init()
#Bombr_train.model_feature_train()
=======
>>>>>>> d92eb4deeb01cea0b68c9a235f81a36e4c0b66d4
>>>>>>> master
