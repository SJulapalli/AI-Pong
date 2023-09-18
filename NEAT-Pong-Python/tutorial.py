import pygame
from pong import Game
import neat
import os

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

def eval_genomes(genomes, config):
    pass

def run_neat(config):
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    p.run(eval_genomes, 50)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

# Running the game
# width, height = 700, 500
# game = Pong_Game(pygame.display.set_mode((width, height)), width, height)

# game.test_ai()