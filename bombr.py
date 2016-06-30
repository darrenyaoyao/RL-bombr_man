"""Main startup file for bomberman"""

from optparse import OptionParser
import sys
import os
import threading

if sys.version_info[0] == 3:
    print 'Python 3 is not supported'
    sys.exit(1)
elif sys.version_info[1] <= 5:
    print 'Python 2.6+ is required'
    sys.exit(1)

import serge.common


parser = OptionParser()
parser.add_option("-f", "--framerate", dest="framerate", default=60, type="int",
                  help="framerate to use for the engine")
parser.add_option("-l", "--log", dest="log", default=40, type="int",
                  help="logging level")
parser.add_option("-p", "--profile", dest="profile", default=False, action="store_true",
                  help="profile the game for speed")
parser.add_option("-d", "--debug", dest="debug", default=False, action="store_true",
                  help="run in debug mode")
parser.add_option("-c", "--cheat", dest="cheat", default=False, action="store_true",
                  help="run in cheat mode - all levels are available right away")
parser.add_option("-m", "--muted", dest="muted", default=False, action="store_true",
                  help="start with all sounds silenced")
parser.add_option("-x", "--music-off", dest="music_off", default=False, action="store_true",
                  help="start with music silenced")
parser.add_option("-S", "--straight", dest="straight", default=True, action="store_true",
                  help="go straight into game, bypassing start screen")
parser.add_option("-s", "--screenshot", dest="screenshot", default=False, action="store_true",
                  help="allow screenshots of the screen by pressing 's' during gameplay")
parser.add_option("-t", "--theme", dest="theme", default='', type='str',
                  help="settings (a=b,c=d) for the theme")
parser.add_option("-M", "--movie", dest="movie", default='', type='str',
                  help="record a movie of the game with the given filename")
parser.add_option("-D", "--drop", dest="drop", default=False, action="store_true",
                  help="drop into debug mode on an unhandled error")
parser.add_option("-H", "--high-score", dest="high_score", default=False, action="store_true",
                  help="recreate the high score table")
parser.add_option("-T", "--test", dest="test", default=False, action="store_true",
                  help="use AI test cases instead of levels")
parser.add_option("-O", "--obser", dest="observation", default="default.obser", type="string",
                  help="observation output")
parser.add_option("-W", "--reward", dest="reward", default="default.reward", type="string",
                  help="reward output")
parser.add_option("-R", "--random", dest="random", default=False, action="store_true",
                  help="random policy")
parser.add_option("-P", "--supervised_policy", dest="supervised_policy", default=False, action="store_true",
                  help="supervised_policy")
parser.add_option("-Q", "--Qmodel", dest="Qmodel", default=False, action="store_true",
                  help="Qmodel")
parser.add_option("-F", "--feature_supervised_policy", dest="feature_supervised_policy", default=False,
                  action="store_true", help="supervised_policy")

observation = [{"action": 0, "flag": 0, "observation": []}]
(options, args) = parser.parse_args()
serge.common.logger.setLevel(options.log)

import game.main
import levels
import game.common


game.common.levels = levels
game.main.main(options, args, observation)
