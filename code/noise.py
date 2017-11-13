import random as random
class Noise(r,q):
    """ Noise class """
    def __init__(self,q):
        self.q = q
        self.r = r
        self.decisions = []
        self.agents = Agents.agents()
        self.empty_locs = PrisonerDilemma.empty_locs()
        self.loc = PrisonerDilemma.loc()

    def choosing_strategy(self):
        if self.q >= random.randomint(0,1):
            self.decisions.append('C')
        else:
            self.decisions.append('D')

    def move_randomly(self):
        self.loc = self.empty_locs[random.randint(0,len(empty_locs)-1)]
        return self.loc


    def both_noise(self):
        return self.choosing_strategy(), self.move_randomly()
