import numpy as np

class Agent:
    score = 0
    color = 0 # white(0)/red(1)/blue(2)/green(3)/yellow(4) (but an agent will never get whilte color)
    """
    Red color indicates that the agent is/was defector
    Blue color indicates that the agent is/was cooperator
    Yellow color indicates that the agent just changed from cooperator to defector
    Green color indicates that the agent just changed from defector to cooperator
    """
    color_lookup = {(1,1):1, (2,2):2, (1,2):3, (2,1):4}

    def __init__(self, initial_state, r=0.05, q=0.05, seed=100):
        self.state = initial_state
        self.best_neighbor = initial_state
        self.color = 1 if initial_state == 1 else 2
        self.r = r # 1-r is probability of imitation
        self.q = q # q is spontaneously cooperate, 1-q is spontaneously defect
        self.seed = seed

    def update_best_performing_neighbor(self, code):
        self.best_neighbor = code

    def choose_next_state(self):
        prev_state = self.state
        self._seed()
        if np.random.rand() < 1-self.r:
            self.state = self.best_neighbor
        else:
            # with probability q, cooperate, else defect
            self._seed()
            self.state = 2 if np.random.rand() < self.q else 1
        # Update color
        self.color = self.color_lookup[(prev_state, self.state)]

    def _seed(self):
        np.random.seed(self.seed)
        self.seed += 1


class SocialAgent(Agent):
    def update_average_neighbor_weights(self, avg_weights):
        self.average_neighbor_weights = avg_weights

    def choose_next_state(self):
        prev_state = self.state
        # With probabilty of trust, cooperate else defect
        self._seed()
        self.state = 2 if np.random.rand() < self.average_neighbor_weights else 1
        # Update color
        self.color = self.color_lookup[(prev_state, self.state)]


class AgentBiulder():
    """constructor for agents, allows differentiation more easily"""
    def __init__(self, loc, state, constructor=Agent, parameters=None):
        pass
