import pygame
from pong import Game
import neat
import os

class PongGame:

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
        
        pygame.quit()

    def train_ai(self, genome1, genome2, config):
        net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
        net2 = neat.nn.FeedForwardNetwork.create(genome2, config)

        run = True

        while run:
            # Handles game quit
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()

            # Activate the Neural Networks with the wanted input nodes
            output1 = net1.activate((self.left_paddle.y, self.ball.y, abs(self.left_paddle.x - self.ball.x)))
            decision1 = output1.index(max(output1))

            output2 = net1.activate((self.right_paddle.y, self.ball.y, abs(self.right_paddle.x - self.ball.x)))
            decision2 = output2.index(max(output2))

            if decision1 == 1:
                self.game.move_paddle(True, True)
            elif decision1 == 2:
                self.game.move_paddle(True, False)
                
            if decision2 == 1:
                self.game.move_paddle(False, True)
            elif decision2 == 2:
                self.game.move_paddle(False, False)

            # print(output1, output2)

            # Handles game loop and display
            game_info = self.game.loop()

            self.game.draw()
            pygame.display.update()

            # If a paddle misses a ball, we stop testing it
            if game_info.left_score >= 1 or game_info.right_score >= 1 or game_info.left_hits > 50:
                self.calculate_fitness(genome1, genome2, game_info)
                break

    def calculate_fitness(self, genome1, genome2, game_info):
        genome1.fitness += game_info.left_hits
        genome2.fitness += game_info.right_hits

def eval_genomes(genomes, config):
    width, height = 700, 500
    window = pygame.display.set_mode((width, height))

    # Test every genome against every other genome
    for i, (genome_id1, genome1) in enumerate(genomes):
        if i == len(genomes) - 1:
            break

        genome1.fitness = 0
        for (genome_id2, genome2) in genomes[i + 1:]:
            genome2.fitness = 0 if genome2.fitness == None else genome2.fitness
            game = PongGame(window, width, height)
            game.train_ai(genome1, genome2, config)

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
    run_neat(config)