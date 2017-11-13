import numpy as np

class Agent():
    """Agent"""

    def __init__(self, loc, state, params=None):
        self.loc = loc # location in environment - (row, column)
        if params == None:
            self.intelligence = 1
            pass
        else:
            # put prameters into instance
            pass
        self.state = state # 'C' or 'D'

    def imitate(self, neighbors):
        pass

    def migrate(self, locs, type='random'):
        """ migration type is random, bias_res, bias_loc, or bias_both"""
        pass

    def play_pd(self, neighbors):
        pass

    def step(self, neighbors):
        """noise reaction, interation w/ neighbors"""
        pass

class AgentBiulder():
    """constructor for agents, allows differentiation more easily"""
    def __init__(self, loc, state, constructor=Agent, parameters=None):
        pass