import threading, time

import Globals, pygame, os, warnings

from builders import EnvironmentBuilder, ModelBuilder
from trackers.GameTracker import GameTracker

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
        self.size = self.weight, self.height = 160, 210

    def on_init(self):
        Globals.init()
        EnvironmentBuilder.init()
        ModelBuilder.init()
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
                self._running = False

            # HARD RESET
            if event.key == pygame.K_q:
                pass
                # ModelBuilder.init()
                # self.game_session_init()

            # SOFT RESET
            if event.key == pygame.K_r:
                # print('SOFT RESET')
                # self.game_session_init()
                self._start_trial = False

    def game_session_init(self):
        self._learning_thread = threading.Thread(target=Globals.model.learn, args=(360000,))
        self._learning_thread.start()

    def on_loop(self):
        pass
        # action, states = Globals.model.predict(Globals.obs)
        # Globals.obs, rewards, dones, info = Globals.env.step(action)

    def on_render(self):
        rgb = Globals.env.render(mode='rgb_array')
        image = np.transpose(rgb, (1, 0, 2))
        pygame.pixelcopy.array_to_surface(self._display_surf, image)
        pygame.display.flip()
        # while Globals.paus_game:
        #     pass
        # if Globals.image is not None:
        #     pygame.pixelcopy.array_to_surface(self._display_surf, Globals.image)
        #     pygame.display.flip()

    def on_cleanup(self):
        self._learning_thread.kill()
        pygame.quit()

    def on_execute(self):
        if self.on_init() == False:
            self._running = False

        self.game_session_init()

        while self._running:
            for event in pygame.event.get():
                self.on_event(event)

            self.on_loop()
            self.on_render()
        self.on_cleanup()


if __name__ == "__main__":
    theApp = App()
    theApp.on_execute()
