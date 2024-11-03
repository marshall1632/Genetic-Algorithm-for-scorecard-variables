from random import random


class Individual():
    def __init__(self, space, price, space_limit, generation=0):
        self.space = space
        self.price = price
        self.space_limit = space_limit
        self.score_evalution =0
        self.used_space = 0
        self.generation = generation
        self.chromosome = []

        for i in range(len(space)):
            if random() < 0.5:
                self.chromosome.append('0')
            else:
                self.chromosome.append('1')

    def fittness(self):
        score = 0
        sum_space = 0
        for i in range(len(self.chromosome)):
            if self.chromosome[i] == '1':
                sum_space += self.space[i]
        if sum_space > self.space_limit:
            score = 1
        self.score_evalution = score
        self.used_space = sum_space
