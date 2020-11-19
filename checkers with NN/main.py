# Assets: https://techwithtim.net/wp-content/uploads/2020/09/assets.zip
from World import *
from minimax.algorithm import minimax
from Player import Player
FPS = 60

red_brain = NN(32, [(40, 'RELU'), (40, 'RELU'), (20, 'RELU')], (1, 'SIGMOID'))
white_brain = NN(32, [(40, 'RELU'), (40, 'RELU'), (20, 'RELU')], (1, 'SIGMOID'))

red_player = Player(red_brain, 1)
white_player = Player(red_brain, 0)

WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Checkers')

earth = World(red_player, white_player, WIN)
#print(WHITE, RED)
earth.running()


def get_row_col_from_mouse(pos):
    x, y = pos
    row = y // SQUARE_SIZE
    col = x // SQUARE_SIZE
    return row, col


def main():
    print("goin")
    run = True
    clock = pygame.time.Clock()
    game = Game(WIN)

    while run:
        clock.tick(FPS)

        if game.turn == WHITE:
            value, new_board = minimax(game.get_board(), 2, WHITE, game)
            game.ai_move(new_board)

        if game.winner() != None:
            print("koniec")
            print(game.winner())
            run = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                row, col = get_row_col_from_mouse(pos)
                game.select(row, col)

        game.update()

    pygame.quit()
