from bombrtrain import bombrtrain
import argparse

parser = argparse.ArgumentParser(description='Bombr Game')

parser.add_argument('-O', action="store", dest='obser')
parser.add_argument('-W', action="store", dest='reward')

Bombr_train = bombrtrain(parser.parse_args())
Bombr_train.models_policy_train()
