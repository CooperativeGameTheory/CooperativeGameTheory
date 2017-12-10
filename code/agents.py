import numpy as np

class Agent():
    """Agent"""

    def __init__(self, loc, state, array, params=None):
        self.loc = loc # location in environment - (row, column)
        if params == None:
            self.loc_options = 1
        else:
            # put prameters into instance
            pass
        self.score = 0
        self.status = state # 'C' or 'D'
        self.color = None

    def imitate(self, neighbors):
        best_neighbor = max(neighbors, key= lambda x: x.score)
        self.status = best_neighbor.status

    def step(self, neighbors):
        """noise reaction, interation w/ neighbors"""
        pass
