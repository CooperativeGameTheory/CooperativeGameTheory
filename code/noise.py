import random as random
class Noise(r,q):
    """ Noise class """
    def __init__(self,loc, q):
        self.q = q
        self.r = r
        self.decisions = []
        self.agents = Agents.agents()
        self.empty_locs = PrisonerDilemma.empty_locs()
        self.loc = loc

    def changing_strategy(self):
        "strategy changes from imitation to randomly choosing to cooperate or defect"
        if self.q >= random.randomint(0,1):
            self.decisions.append('C')
        else:
            self.decisions.append('D')

    def migrate_randomly(self):
        "migration is completely random"
        self.loc = self.empty_locs[random.randint(0,len(empty_locs)-1)]
        return self.loc


    def both_noise(self):
        "agent will migrate randomly and choose their strategy with a biased random"
        return self.changing_strategy(), self.migrate_randomly()
