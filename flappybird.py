from math import remainder
from operator import truediv
import pygame
import random
import neat
import time
import os
pygame.font.init()

WIN_WIDTH = 500
WIN_HEIGHT = 800

BIRD_IMGS = [pygame.transform.scale2x(pygame.image.load(os.path.join("photo", "bird1.png"))), pygame.transform.scale2x(
    pygame.image.load(os.path.join("photo", "bird2.png"))), pygame.transform.scale2x(pygame.image.load(os.path.join("photo", "bird2.png")))]
PIPE_IMGS = pygame.transform.scale2x(
    pygame.image.load(os.path.join("photo", "pipe.png")))
BASE_IMGS = pygame.transform.scale2x(
    pygame.image.load(os.path.join("photo", "base.png")))
BG_IMGS = pygame.transform.scale2x(
    pygame.image.load(os.path.join("photo", "bg.png")))

STAT_FONT = pygame.font.SysFont("comicsans", 50)


class Bird:
    IMG = BIRD_IMGS
    MAX_ROTATION = 25  # how much the bird gonna go up or down
    ROT_VEL = 20  # how much the game gonna go rotate
    ANIMATION_TIME = 5  # The changing time for different bird image

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0  # start at 0 before any movement
        self.tick_count = 0  # count the time the bird move up or down
        self.vel = 0  # have not moved
        self.height = self.y
        self.img_count = 0  # keep track which animation to show image
        self.img = self.IMG[0]

    def jump(self):
        self.vel = -10.5  # negative bc moving upward (x,y)
        self.tick_count = 0  # keep track the last jump
        self.height = self.y

    def move(self):
        self.tick_count += 1
        # when move, base on the tickcount will decide to move up or down
        d = self.vel*self.tick_count + 1.5*self.tick_count**2

        if d >= 16:  # keep in control, not moving too fast
            d = 16
        if d < 0:
            d -= 2

        self.y = self.y + d  # move up or down

        if d < 0 or self.y < self.height + 50:  # decision to tilt
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:  # moving downward
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL  # allow to moving down 90de

    def draw(self, win):
        self.img_count += 1

        if self.img_count < self.ANIMATION_TIME:
            self.img = self.IMG[0]
        elif self.img_count < self.ANIMATION_TIME*2:
            self.img = self.IMG[1]
        elif self.img_count < self.ANIMATION_TIME*3:
            self.img = self.IMG[2]
        elif self.img_count < self.ANIMATION_TIME*4:
            self.img = self.IMG[1]
        elif self.img_count == self.ANIMATION_TIME*4 + 1:
            self.img = self.IMG[0]
            self.img_count = 0

        if self.tilt <= -80:
            self.img = self.IMG[1]
            self.img_count = self.ANIMATION_TIME*2

        rotated_image = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rotated_image.get_rect(
            center=self.img.get_rect(topleft=(self.x, self.y)).center)
        win.blit(rotated_image, new_rect.topleft)

    def get_mask(self):
        return pygame.mask.from_surface(self.img)


class Pipe:
    GAP = 200
    VEL = 5

    def __init__(self, x):  # not y because it gonna be random
        self.x = x
        self.height = 0
        self.gap = 100  # between top and bottom

        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMGS, False, True)
        self.PIPE_BOTTOM = PIPE_IMGS

        self.passed = False
        self.set_height()

    def set_height(self):  # defind to top and bottom and the gap
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height()  # figure out where to draw
        self.bottom = self.height + self.GAP

    def move(self):
        self.x -= self.VEL

    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        # findout how far the bird from the bottom pipe
        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        # findout how far the bird from the bottom pipe
        t_point = bird_mask.overlap(top_mask, top_offset)

        if t_point or b_point:
            return True

        return False


class Base:
    VEL = 5
    WIDTH = BASE_IMGS.get_width()
    IMG = BASE_IMGS

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL

        if self.x1 + self.WIDTH < 0:  # check if off the window to back
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:  # check if off the window to back
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))


def draw_window(win, birds, pipes, base, score):
    win.blit(BG_IMGS, (0, 0))
    for pipe in pipes:
        pipe.draw(win)

    text = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))

    base.draw(win)

    for bird in birds:
        bird.draw(win)

    pygame.display.update()


def main(genomes, config):
    nets = []
    ge = []
    birds = []

    for _, g in genomes:  # setting up neuron network
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        birds.append(Bird(230, 350))
        g.fitness = 0
        ge.append(g)

    base = Base(730)
    pipes = [Pipe(700)]
    score = 0
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()
    run = True

    while run:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.display.quit()
                pygame.quit()
                exit()

        pipe_ind = 0
        if len(birds) > 0:
            # if passes the pipes
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1
        else:  # if no bird left, stop
            run = False
            break

        for x, bird in enumerate(birds):
            bird.move()
            ge[x].fitness += 0.1

            output = nets[x].activate((bird.y), abs(
                bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom))

            if output[0] > 0.5:  # the first element of the list
                bird.jump

        add_pipe = False
        rem = []

        for pipe in pipes:
            for x, bird in enumerate(birds):
                if pipe.collide(bird):  # remove the bird from the list
                    ge[x].fitness -= 1
                    birds.pop(x)
                    nets.bird(x)
                    ge.pop(x)

                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True

            pipe.move()

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

        if add_pipe:
            score += 1
            for g in ge:
                g.fitness += 5
            pipes.append(Pipe(WIN_WIDTH))

        for r in rem:
            pipes.remove(r)

        for x, bird in enumerate(birds):
            if bird.y + bird.img.get_height() >= 730 or bird.y < 0:
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)

        # bird.move()
        base.move()

        draw_window(win, BIRD_IMGS, pipes, base, score)


def run(config_path):
    config = neat.config.Congig(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticReporter()
    p.add_reporter(stats)

    winner = p.run(main, 50)


if __name__ == "main":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)
