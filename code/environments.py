import numpy as np
class Environment:
    """ Environment """
    def __init__(self, n=49):
        # Encode
        # Empty location as 0



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




