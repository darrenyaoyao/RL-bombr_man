import bombrtrain
import argparse

parser = argparse.ArgumentParser(description='Bombr Game')

parser.add_argument('-O', action="store", dest='obser')
parser.add_argument('-W', action="store", dest='reward')

print (parser.parse_args())
