import numpy as np
from scipy.signal import correlate2d


class Agent:
    score = 0
    color = 0 # white(0)/red(1)/blue(2)/green(3)/yellow(4) (but an agent will never get whilte color)
    color_lookup = {(1,1):1, (2,2):2, (1,2):3, (2,1):4}

    def __init__(self, env, location, initial_state, r, q):
        self.env = env
        self.state = initial_state
        self.r = r # 1-r is probability of imitation
        self.q = q # q is spontaneously cooperate, 1-q is spontaneously defect

    def update_best_performing_neighbor(self, code):
        self.best_neighbor = code

    def choose_next_state(self):
        prev_state = self.state
        if np.random.rand() < 1-self.r:
            self.state = self.best_neighbor
        else:
            # with probability q, cooperate, else defect
            self.state = 2 if np.random.rand() < self.q else 1
        # Update color
        self.color = color_lookup[(prev_state, self.state)]


class Environment:
    """ Environment
    L = length of one side of 2D locations
    n = number of agents
    seed = seed for controlling randomness
    M = radius of migration
    """

    # <Encoding of env array>
    # Empty = 0
    # Defector = 1
    # Cooperator = 2
    # kernel = np.array([0x10**0, 0x10**1,    0x10**2],
    #                   [0x10**3, 0x44444444, 0x10**4],
    #                   [0x10**5, 0x10**6,    0x10**7], dtype=np.uint64)

    # kernel is used for simulating PD
    kernel = np.array([[      0, 0x10**0,       0],
                       [0x10**1,  0x4444, 0x10**2],
                       [      0, 0x10**3,       0]], dtype=np.uint64)

    # kernel2 is used for finding person with highest score
    # Maximum score an agent can achieve in one round is 52 (when T=13)
    kernel2 = np.array([[     0,  100**1,       0],
                        [100**2,       1,  100**3],
                        [     0,  100**4,       0], dtype=np.uint64)
    rule = {10:self.R, 9:self.S, 6:self.T, 5:self.P, 0:0, 1:0, 2:0, 4:0, 8:0}

    def __init__(self, L=49, seed=0, R=10, S=0, T=13, P=1):
        self.size = size = (L, L)
        # Encode 'env' matrix to have
        # Empty location as 0, Defector as 1, Cooperator as 2
        self.env = np.zeros(size, dtype=np.uint64)

        # Register vectorized count score function and get state function
        self.vcountScore = np.vectorize(self._countScore)
        self.vgetState = np.vectorize(lambda x: x.state if x is not None else 0)
        self.vfindBest = np.vectorize(self._findBest)

        # Make a matrix that holds Agent objects
        self.agents = np.full(size, None)

        # Register seed
        self.seed = seed

        # Register R,S,T,P
        self.R, self.S, self.T, self.P = R, S, T, P

    def place_agents(self):
        """
        Places agents randomly and leave 50% empty sites
        You can override this function by inheriting Environment class
        and implementing your own place_agents(self) function
        """
        L, _ = self.size
        all_indices = 
        self._seed()
        chosen_indices = np.random.choice(L*L, L*L//2, replace=False)




    def playRound(self):
        """
        Simulates Prisoner's Dillema, calculates score for each cell, and updates agents
        """
        # Default correlate2d handles boundary with 'fill' option with 'fillvalue' zero.
        c = correlate2d(self.env, self.kernel, mode="same")

        # Each digit (in hex) of the number (in each cell) can be either:
        # 0 : both cell is empty
        # 1,2,4,8 : one of the cell is empty
        # 5 : both cell defected
        # 6 : middle cell Defected, the other cell Cooperated
        # 9 : middle cell Cooperated, the other cell Defected
        # A : both cell cooperated

        # Gives scores for each cell
        scores = self.vcountScore(c) # n x n nparray

        # Find best for each cell
        surrounding = correlate2d(scores, self.kernel2, mode="same")
        best = self.vfindBest(surrounding)

        # Update agents
        for idx, agent in np.ndenumerate(self.agents):
            if agent is None:
                continue
            x, y = idx
            deltaX, deltaY = best[idx]
            best_neighbor_state = self.env[x+deltaX, y+deltaY]
            agent.update_best_performing_neighbor(best_neighbor_state)
            agent.choose_next_state()

        self.update_env()

    def migrate(self):
        """
        Randomly migrate to different empty location
        """
        n, _ = self.size
        env = self.env
        agents = self.agents

        # Probability of relocation
        p = 0.05

        # Choose migrating agents
        migrator_indices = np.argwhere(self.env > 0)
        self._seed()
        r = np.random.rand(migrator_indices[0])
        choices = migrator_indices[r<p]
        migrator_indices = migrator_indices[choices, :]
        self._seed()
        if len(migrator_indices):
            np.random.shuffle(migrator_indices)

        # numpy array of empty indices of shape (k, 2)
        empty_indices = np.argwhere(self.env == 0)

        # for each migrating agent, choose random destination
        for source in migrator_indices:
            source = tuple(source)
            i = np.random.randint(len(empty_indices))
            dest = tuple(empty_indices[i])

            # move
            agents[dest] = agents[source]
            agents[source] = None
            empty_indices[i] = source

    def update_env(self):
        """ Based on the locations of agents, update the env """
        self.env = self.vgetState(self.agents)



    # private functions
    def _countScore(self, num):
        total = 0
        while num > 0:
            digit = num % 0x10
            total += self.rule[digit]
            num //= 0x10
        return total

    def _seed(self):
        np.seed(self.seed)
        self.seed += 1

    def _findBest(self, num):
        """
        Given number calculated by correlate2d,
        find the location (relative to self)
        where it has the highest score
        """
        middle, num = num % 100, num//100
        top, num = num % 100, num//100
        left, num = num % 100, num//100
        right, num = num % 100, num//100
        bottom = num % 100
        d = {(0,0):middle, (0, 1):top, (-1,0):left, (1,0):right, (0,-1):bottom}
        return max(d, key=d.get)


#class PrisonerDilemma:
#    def __init__(self, T, R, P, S):
#        """
#        T : Temptation payoff
#        R : Reward for cooperating
#        P : Punishment payoff for both defect
#        S : Sucker's payoff
#        (T > R > P > S) condition indicates prisoner's dillema condition
#        (T > R > P > S) condition indicates snowdrift game
#        """
#        rules = {}
#        rules[('C', 'C')] = R, R
#        rules[('D', 'C')] = T, S
#        rules[('C', 'D')] = S, T
#        rules[('D', 'D')] = P, P
#        self.rules = rules
#
#    def playgame(self, agentA, agentB):
#        A, B = agentA.status, agentB.status
#        rules[(A,B)]
#
#
#

