import threading

import Globals
from builders import ModelBuilder, EnvironmentBuilder


class LearningThread(threading.Thread):

    def __init__(self,  *args, **kwargs):
        super(LearningThread, self).__init__(*args, **kwargs)
        self._stop_event = threading.Event()

    def init_model(self):
        EnvironmentBuilder.init()
        ModelBuilder.init()

    def run(self):
        Globals.exit_learning = False
        Globals.pause_game = True
        Globals.model.learn(20000)

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()
