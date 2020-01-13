import Globals
import time
import json
import os

import pandas as pd

from datetime import datetime, date


class GameTracker:
    files_n = None

    results_dir = 'tensor'
    results_session_dir = 'tensor/DQN_{0}'

    results_model = 'model'
    results_file = 'results.csv'

    start_timestap = None
    end_timestap = None

    game_number = 0

    def __init__(self):
        if not os.path.exists(self.results_dir):
            self.files_n = 1
            os.makedirs(self.results_dir)
        else:
            self.files_n = len(os.listdir(self.results_dir)) + 1

    def save_results(self):
        dir_name = self.results_session_dir.format(self.files_n)
        results = pd.DataFrame(Globals.results_list).drop_duplicates()
        results.to_csv(dir_name + '/results.csv', index=False)
        Globals.model.save(dir_name + '/model.h5')
        self.files_n += 1