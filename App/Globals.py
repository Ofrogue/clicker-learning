def init():
    global reward
    global score
    global given_rewards

    global paus_game

    global model
    global env

    global image

    reward = 0.0
    given_rewards = 0
    score = 0

    paus_game = True

    model = None
    env = None
    image = None
