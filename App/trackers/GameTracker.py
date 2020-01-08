import Globals
import time
import json
import os

import pandas as pd

from datetime import datetime, date


class GameTracker:
    results_dir = 'results/{0}'
    results_file_name = None

    start_timestap = None
    end_timestap = None

    game_number = 0

    def __init__(self):
        self.results_dir = self.results_dir.format(date.today())
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        self.results_file_name = '{0}.csv'.format(len(os.listdir(self.results_dir)))

    def start(self):
        Globals.score = 0
        Globals.given_rewards = 0
        self.game_number += 1
        self.start_timestap = datetime.now()

    def end(self):
        self.end_timestap = datetime.now()

    def get_results(self):
        return {
                'game': self.game_number,
                'start timestap': '{0}'.format(self.start_timestap),
                'end timestap': '{0}'.format(self.end_timestap),
                'time delta': '{0}'.format(self.end_timestap - self.start_timestap),
                'score': Globals.score,
                'number of rewards': Globals.given_rewards
            }

    def save_results(self):
        results = pd.DataFrame([self.get_results()])
        file_path = self.results_dir + '/' + self.results_file_name
        if os.path.exists(file_path):
            results.to_csv(file_path, mode='a', index=False, header=False)
        else:
            results.to_csv(file_path, index=False)
