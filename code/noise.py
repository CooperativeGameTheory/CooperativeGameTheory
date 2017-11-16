import random as random
class Noise(Environment):
    """ Noise class """
    def __init__(self,loc,q=.05,r=.05):
        super(Environment, self).__init__( L=49, seed=0, R=10, S=0, T=13, P=1, **config)
        self.q = q #will choose to cooperate with probabiility 1-q
        self.r = r

    def changing_strategy(self):
        "strategy changes from imitation to randomly choosing to cooperate or defect"
        "noah implemented this in his class"
        pass

    def migrate(self):
        "migration is completely random - taking Tony's code because this is random migration"
        n, _ = self.size
        env = self.env
        agents = self.agents

        # Probability of random relocation
        p = .05

        # Choose migrating agents
        migrator_indices = np.argwhere(self.env > 0)
        self._seed()
        r = np.random.rand(migrator_indices.shape[0])
        migrator_indices = migrator_indices[r<p]
        if len(migrator_indices):
            self._seed()
            np.random.shuffle(migrator_indices)

        # numpy array of empty indices of shape (k, 2)
        empty_indices = np.argwhere(self.env == 0)

        # for each migrating agent, choose random destination
        for source in migrator_indices:
            source = tuple(source)
            self._seed()
            i = np.random.randint(len(empty_indices))
            dest = tuple(empty_indices[i])

            # move
            agents[dest] = agents[source]
            agents[source] = None
            empty_indices[i] = list(source)
