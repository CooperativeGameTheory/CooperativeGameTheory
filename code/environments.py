import numpy as np
from scipy.signal import correlate2d



class Environment:
    """ Environment """
    # <Encoding>
    # Empty = 0
    # Defector = 1
    # Cooperator = 2
    kernel = np.array([0x10**0, 0x10**1,    0x10**2],
                      [0x10**3, 0x44444444, 0x10**4],
                      [0x10**5, 0x10**6,    0x10**7], dtype=np.uint64)
    rule = {10:self.R, 9:self.S, 6:self.T, 5:self.P, 0:0, 1:0, 2:0, 4:0, 8:0}
    def __init__(self, n=49):
        size = (n, n)
        # Encode 'env' matrix to have
        # Empty location as 0, Defector as 1, Cooperator as 2
        self.env = np.zeros(size, dtype=np.uint64)

        # Register vectorized count score function
        self.vcountScore = np.vectorize(self._countScore)

        # Make a matrix that holds Agent objects
        self.agents = np.full(size, None)


    def playRound(self):
        # default correlate2d handles boundary with 'fill' option with 'fillvalue' zero.
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

    def migrate(self):
        # TODO: Logic to select migrating agents

        empty_indices = np.argwhere(self.env == 0)


    # private functions
    def _countScore(self, num):
        total = 0
        while num > 0:
            digit = num % 0x10
            total += self.rule[digit]
            num //= 0x10
        return total



class PrisonerDilemma:
    def __init__(self, T, R, P, S):
        """
        T : Temptation payoff
        R : Reward for cooperating
        P : Punishment payoff for both defect
        S : Sucker's payoff
        (T > R > P > S) condition indicates prisoner's dillema condition
        (T > R > P > S) condition indicates snowdrift game
        """
        rules = {}
        rules[('C', 'C')] = R, R
        rules[('D', 'C')] = T, S
        rules[('C', 'D')] = S, T
        rules[('D', 'D')] = P, P
        self.rules = rules

    def playgame(self, agentA, agentB):
        A, B = agentA.status, agentB.status
        rules[(A,B)]




