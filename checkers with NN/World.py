import time

from NeuralNetwork import *
from checkers.game import *
from minimax.algorithm import get_all_moves


class World:
    def __init__(self, red_player, white_player, window):
        self.red_player = red_player
        self.white_player = white_player
        self.window = window
        self.game = Game(window)
        self.generation = 0
        self.delay = 1
        self.showing = True

    def possible_moves(self, color):
        return get_all_moves(self.game.get_board(), color)

    def binary_data(self, color):

        states = self.possible_moves(color)
        data = []

        if color == WHITE:
            enemy = RED
        else:
            enemy = WHITE

        for state in states:
            board = state.board
            binary_board = []

            for i in range(8):
                if i % 2 == 0:  # od góry
                    for j in range(1, 8, 2):
                        if board[i][j] == 0:
                            binary_board.append(0)
                        else:
                            if board[i][j].color == color:
                                binary_board.append(1)
                            if board[i][j].color == enemy:
                                binary_board.append(-1)

                if i % 2 == 1:
                    for j in range(0, 7, 2):
                        if board[i][j] == 0:
                            binary_board.append(0)
                        else:
                            if board[i][j].color == color:
                                binary_board.append(1)
                            if board[i][j].color == enemy:
                                binary_board.append(-1)

            data.append(np.asarray(binary_board))

        return data

    def running(self):
        run = True
        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False

            if self.game.winner() is not None:
                if self.game.winner() == RED:
                    self.red_player.brain.reward = REWARD
                    self.white_player.brain.reward = PUNISHMENT
                else:
                    self.red_player.brain.reward = PUNISHMENT
                    self.white_player.brain.reward = REWARD

                self.red_player.brain.learning()
                self.white_player.brain.learning()

                print(self.generation, self.game.winner())

                self.game.reset()
                self.generation += 1
                

            if self.game.turn == RED:
                # print("red")
                states = self.possible_moves(RED)
                binary_moves = self.binary_data(RED)
                self.game.ai_move(states[self.red_player.pick_move(binary_moves)])

            else:
                # print("white")
                states = self.possible_moves(WHITE)
                binary_moves = self.binary_data(WHITE)
                self.game.ai_move(states[self.white_player.pick_move(binary_moves)])

            if self.showing:
                self.game.update()
                time.sleep(self.delay)
        pygame.quit()
