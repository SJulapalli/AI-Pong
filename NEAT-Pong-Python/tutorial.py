import pygame
from pong import Game

class Pong_Game:

    # Define game window details
    def __init__(self, window, width, height):
        self.game = Game(window, width, height)
        self.left_paddle = self.game.left_paddle
        self.right_paddle = self.game.right_paddle
        self.ball = self.game.ball

    def test_ai(self):
        run = True
        clock = pygame.time.Clock()

        while run:

            # Handles game speed
            clock.tick(60)

            # Handles game exit
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break

            # Handles player input
            keys = pygame.key.get_pressed()

            if keys[pygame.K_UP]:
                self.game.move_paddle(True, True)
            elif keys[pygame.K_DOWN]:
                self.game.move_paddle(True, False)

            # Handles game loop and display
            self.game.loop()
            self.game.draw()
            pygame.display.update()