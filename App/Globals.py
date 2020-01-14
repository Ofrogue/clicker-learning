from LearningThread import LearningThread


def init():
    global reward
    global blocks
    global score
    global results_list
    global given_rewards

    global steps

    global loading
    global pause_game
    global exit_learning

    global env
    global model

    reward = 0.0
    given_rewards = 0
    blocks = 0
    score = 0
    steps = LearningThread.iteration
    results_list = []

    loading = True
    pause_game = True
    exit_learning = False

    env = None
    model = None
