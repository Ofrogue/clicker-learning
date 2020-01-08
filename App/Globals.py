def init():
    global reward
    global score
    global given_rewards

    global paus_game
    global exit_learning

    global reset_env

    global env
    global model

    reward = 0.0
    given_rewards = 0
    score = 0

    paus_game = True
    exit_learning = False
    reset_env = False

    env = None
    model = None
