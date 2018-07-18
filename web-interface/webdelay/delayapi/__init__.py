from flask import Flask
from .app import app
from . import flightDelayPred
import logging as lg
#from . import config

## VARIABLES


log = lg.getLogger('werkzeug')
log.setLevel(lg.DEBUG)

flightDelayPred.init()
   