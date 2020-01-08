import Globals, pygame, os, warnings, threading, time, cv2

from trackers.GameTracker import GameTracker
from LearningThread import LearningThread

import tensorflow as tf
import numpy as np

warnings.filterwarnings("ignore")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ["KMP_AFFINITY"] = "none"


# game_tracker = GameTracker()
# game_tracker.start()
#
# for _ in range(1000):
#     env.render()
#     action, states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     time.sleep(0.05)
#
#     if info[0]['ale.lives'] == 0:
#         game_tracker.end()
#         game_tracker.save_results()
#         game_tracker.start()


class App:
    def __init__(self):
        self._game_tracker = GameTracker()
        self._running = True
        self._display_surf = None
        self._learning_thread = None
        self.size = self.weight, self.height = 320, 420

    def on_init(self):
        Globals.init()
        pygame.init()
        self._display_surf = pygame.display.set_mode(self.size, pygame.HWSURFACE | pygame.DOUBLEBUF)
        self._running = True

    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                Globals.given_rewards += 1
                Globals.reward = 1.0

            # START GAME
            if event.key == pygame.K_RETURN:
                Globals.paus_game = False

            # QUIT
            if event.key == pygame.K_ESCAPE:
                Globals.exit_learning = True
                self._running = False

            # HARD RESET
            if event.key == pygame.K_q:
                Globals.exit_learning = True
                self._learning_thread.stop()

            # SOFT RESET
            # if event.key == pygame.K_r:
            #     Globals.reset_env = True

    def game_session_init(self):
        self._learning_thread = LearningThread()
        self._learning_thread.init_model()
        self._learning_thread.start()

    def on_loop(self):
        print(Globals.model.get_parameters()['deepq/model/action_value/fully_connected_1/weights:0'])

    def on_render(self):
        rgb = Globals.env.render(mode='rgb_array')
        rgb_trans = np.transpose(rgb, (1, 0, 2))
        image_resize = cv2.resize(rgb_trans, (self.height, self.weight))
        pygame.pixelcopy.array_to_surface(self._display_surf, image_resize)
        pygame.display.flip()

    def on_cleanup(self):
        pygame.quit()

    def on_execute(self):
        if self.on_init() == False:
            self._running = False

        self.game_session_init()

        while self._running:

            if self._learning_thread.stopped():
                self.game_session_init()

            for event in pygame.event.get():
                self.on_event(event)

            self.on_loop()
            self.on_render()

        self.on_cleanup()


if __name__ == "__main__":
    theApp = App()
    theApp.on_execute()
