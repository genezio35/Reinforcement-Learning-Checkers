import random


class Player:
    def __init__(self, brain, side):
        self.brain = brain
        self.side = side

    def pick_move(self, moves):
        pool = []
        pool.append(0)
        for move in moves:
            pool.append(pool[-1]+self.evaluate_move(move))

        picker = random.uniform(0, pool[-1])
        for i in range(len(pool)):
            if picker <= pool[i] and picker > pool[i - 1]:
                choice = i - 1
                break



        self.brain.choices.append(0)
        self.brain.data.append(moves[choice])
        return choice

    def evaluate_move(self, move):
        return self.brain.rate(move)
