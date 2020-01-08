import keyboard
import time
import Globals


def reward_checker():
    while True:
        keyboard.wait('space')
        Globals.reward = 1.0
        Globals.given_rewards += 1
