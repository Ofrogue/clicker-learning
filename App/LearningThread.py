import threading

import Globals
from builders import ModelBuilder, EnvironmentBuilder


class LearningThread(threading.Thread):
    iteration = 5000

    def __init__(self,  *args, **kwargs):
        super(LearningThread, self).__init__(*args, **kwargs)
        self._stop_event = threading.Event()

    def init_model(self):
        EnvironmentBuilder.init()
        ModelBuilder.init()

    def run(self):
        Globals.exit_learning = False
        Globals.pause_game = True
        Globals.model.learn(self.iteration)

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()
