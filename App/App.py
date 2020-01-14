import Globals, pygame, os, warnings, threading, time, cv2

from trackers.GameTracker import GameTracker
from LearningThread import LearningThread

import tensorflow as tf
import numpy as np

warnings.filterwarnings("ignore")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ["KMP_AFFINITY"] = "none"

game_time = 600


class App:
    def __init__(self):
        self._game_tracker = GameTracker()
        self._running = True
        self._main_display = None
        self._game_srf = None
        self._timer_srf = None
        self._timer_font = None
        self._learning_thread = None
        self.timer = game_time
        self.fps = 60
        self.current_frame = 0
        self.timer_srf_size = self.timer_srf_width, self.timer_srf_height = 320, 50
        self.game_srf_size = self.game_srf_width, self.game_srf_height = 320, 420
        self.display_size = self.display_width, self.display_height = 320, 470

    def get_time(self):
        minunts = int(self.timer / 60)
        seconds = int(self.timer % 60)
        return '{0:02d}:{1:02d}'.format(minunts, seconds)

    def on_init(self):
        Globals.init()
        pygame.init()
        self._main_display = pygame.display.set_mode(self.display_size, pygame.HWSURFACE | pygame.DOUBLEBUF)

        self._timer_srf = pygame.Surface(self.timer_srf_size)
        self._timer_srf.fill((0, 0, 0))
        self._timer_font = pygame.font.SysFont('Consolas', 30)
        text = self._timer_font.render(self.get_time(), True, (255, 255, 255))
        textRect = text.get_rect()
        textRect.center = (self.timer_srf_width // 2, self.timer_srf_height // 2)
        self._timer_srf.blit(text, textRect)
        self._main_display.blit(self._timer_srf, (0, 0))

        self._game_srf = pygame.Surface(self.game_srf_size)
        self._game_srf.fill((155, 255, 255))
        self._main_display.blit(self._game_srf, (0, 50))

        pygame.display.flip()

        self._running = True

    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                Globals.given_rewards += 1
                Globals.reward = 1.0

            if event.key == pygame.K_p:
                Globals.paus_game = not Globals.paus_game

            # START GAME
            if event.key == pygame.K_RETURN:
                Globals.paus_game = False

            # QUIT
            if event.key == pygame.K_ESCAPE:
                Globals.paus_game = False
                Globals.exit_learning = True
                self._game_tracker.save_results()
                self._running = False

            # HARD RESET
            if event.key == pygame.K_q:
                self.end_session()

    def end_session(self):
        Globals.paus_game = False
        Globals.exit_learning = True
        self._game_tracker.save_results()
        self._learning_thread.stop()
        Globals.paus_game = True

    def game_session_init(self):
        self.timer = game_time
        self._learning_thread = LearningThread()
        self._learning_thread.init_model()
        self._learning_thread.start()

    def on_loop(self):
        if not Globals.paus_game:
            self.current_frame += 1
            if self.current_frame == self.fps:
                self.timer -= 1
                self.current_frame = 0
                if self.timer == 0:
                    self.end_session()


    def on_render(self):
        self._timer_srf.fill((0, 0, 0))
        text = self._timer_font.render(self.get_time(), True, (255, 255, 255))
        textRect = text.get_rect()
        textRect.center = (self.timer_srf_width // 2, self.timer_srf_height // 2)
        self._timer_srf.blit(text, textRect)
        self._main_display.blit(self._timer_srf, (0, 0))

        if Globals.loading:
            pass
            # self._display_surf.fill((255, 255, 255))
            # text_surface, rect = self._loading_font.render("Loading", (0, 0, 0))
            # self._display_surf.blit(text_surface, (40, 250))
            # self._loading_font.render_to(self._display_surf, (40, 350), "Hello World!", (0, 0, 0))
        else:
            rgb = Globals.env.render(mode='rgb_array')
            rgb_trans = np.transpose(rgb, (1, 0, 2))
            image_resize = cv2.resize(rgb_trans, (self.game_srf_height, self.game_srf_width))
            pygame.pixelcopy.array_to_surface(self._game_srf, image_resize)
            self._main_display.blit(self._game_srf, (0, 50))

        pygame.display.update()

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

            pygame.time.Clock().tick(self.fps)

        self.on_cleanup()


if __name__ == "__main__":
    theApp = App()
    theApp.on_execute()
