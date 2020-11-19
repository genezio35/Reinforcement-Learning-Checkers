import pygame

WIDTH, HEIGHT = 800, 800
ROWS, COLS = 8, 8
SQUARE_SIZE = WIDTH // COLS

# rgb
RED = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREY = (128, 128, 128)

FPS = 300
NN_VARIABILITY = 0.5
REWARD = 10
PUNISHMENT = -5
LEARNING_RATE = 0.05
RELUPLUS = 1
RELUMINUS = 0.1
CROWN = pygame.transform.scale(pygame.image.load('assets/crown.png'), (44, 25))
