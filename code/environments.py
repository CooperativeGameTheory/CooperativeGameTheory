import numpy as np
from scipy.signal import correlate2d
import matplotlib.pyplot as plt
from matplotlib import colors, animation


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

    def choose_next_state(self, trust):
        prev_state = self.state
        self._seed()
        if np.random.rand() < 1-self.r:
            self.state = self.best_neighbor
        else:
            # with probability q, cooperate, else defect
            self._seed()
            self.state = 2 if trust > .5 else 1
        # Update color
        self.color = self.color_lookup[(prev_state, self.state)]

    def _seed(self):
        np.random.seed(self.seed)
        self.seed += 1


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
                        [     0,  100**4,       0]], dtype=np.uint64)

    # Things needed for plotting
    cmap = colors.ListedColormap(['white', 'red', 'blue', 'green', 'yellow'])
    bounds=[0,1,2,3,4]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    def __init__(self, L=49, seed=0, R=10, S=0, T=13, P=1, **config):
        self.size = size = (L, L)
        # Encode 'env' matrix to have
        # Empty location as 0, Defector as 1, Cooperator as 2
        self.env = np.zeros(size, dtype=np.uint64)
        # random trust initialization
        self.trust_network = np.random.rand(L*L//2, L*L//2)
        # uniform trust initialization
        # self.trust_network = np.multiply(np.ones((L*L//2, L*L//2,), dtype=np.float64), 1/2)

        # Register vectorized count score function and get state function
        self.vcountScore = np.vectorize(self._countScore)
        self.vgetState = np.vectorize(lambda x: x.state if x is not None else 0)
        self.vgetColor = np.vectorize(lambda x: x.color if x is not None else 0)
        self.vfindBest = np.vectorize(self._findBest)

        # Make a matrix that holds Agent objects
        self.agents = np.full(size, None)

        # a dictionary to hold locations af agents to agent ids
        self.locations = {}

        # Register seed
        self.seed = seed

        # Register R,S,T,P
        self.rule = {10:R, 9:S, 6:T, 5:P, 0:0, 1:0, 2:0, 4:0, 8:0}

        # Register configurations
        self.config = config

    def place_agents(self):
        """
        Places agents randomly and leave 50% empty sites
        You can override this function by inheriting Environment class
        and implementing your own place_agents(self) function
        """
        L, _ = self.size
        all_indices = np.array(list(np.ndindex(self.size)))
        self._seed()
        chosen_indices = np.random.choice(L*L, L*L//2, replace=False)

        for i, idx in enumerate(all_indices[chosen_indices,:]):
            self._seed()
            initial_state = np.random.choice([1,2])
            self._seed()
            self.agents[tuple(idx)] = Agent(initial_state, seed=self.seed)
            self.locations[tuple(idx)] = i+1

        self.update_env()


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

        # Encoded as middle = 0, top = 1, left = 2, right = 3, bottom = 4
        # Note that x is indexed first and y is indexed last (x is vertical, y is horizontal)
        tuple_lookup = {0:(0,0), 1:(-1,0), 2:(0,-1), 3:(0,1), 4:(1,0)}

        # Update agents
        for idx, agent in np.ndenumerate(self.agents):
            if agent is None:
                continue
            x, y = idx
            deltaX, deltaY = tuple_lookup[best[idx]]
            best_neighbor_state = self.env[x+deltaX, y+deltaY]
            agent.update_best_performing_neighbor(best_neighbor_state)
            agent.choose_next_state(self.getTrustAverage((x,y)))


        if self.config.get("migrate"):
            self.migrate()

        self.update_env()

        return scores

    def migrate(self):
        """
        Randomly migrate to different empty location
        You can override this function by inheriting Environment class
        and implementing your own migrate(self) function
        """
        n, _ = self.size
        env = self.env
        agents = self.agents

        # number of relocation: proportional to p
        p = 0.05
        L, _ = self.size
        reloc_num = int((p * L * L) // 2)

        # Choose migrating agents
        migrator_indices = np.argwhere(self.env > 0)
        trust_at_loc = {}
        for index in migrator_indices:
            trust_at_loc[tuple(index)] = self.getTrustAverage(tuple(index))
        sorted_trust_at_loc = sorted(trust_at_loc, key=trust_at_loc.get)
        migrator_indices = sorted_trust_at_loc[0:reloc_num]
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
            ident = self.locations[source]
            del self.locations[source]
            self.locations[dest] = ident
            agents[source] = None
            empty_indices[i] = list(source)

    def update_env(self):
        """ Based on the locations of agents, update the env """
        self.env = self.vgetState(self.agents)


    def visualize(self, show=True):
        self.img = plt.imshow(self.vgetColor(self.agents), cmap=self.cmap, norm=self.norm)
        if show:
            plt.show()

    def animate(self, frames=200, interval=50):
        """
        frames : number of frames to draw
        interval : time between frames in ms
        200 frames with 50 interval should take 10 seconds
        """
        def step(i):
            if i>0:
                self.playRound()
            a = self.vgetColor(self.agents)
            self.img.set_array(a)
            return (self.img,)

        fig = plt.figure()
        self.visualize(show=False)
        anim = animation.FuncAnimation(fig, step, frames=frames, interval=interval)
        return anim

    def debug(self, x, y):
        c = correlate2d(self.env, self.kernel, mode="same")
        scores = self.vcountScore(c) # n x n nparray
        surrounding = correlate2d(scores, self.kernel2, mode="same")
        best = self.vfindBest(surrounding)
        chopper = slice(max(0, x-1), x+2), slice(max(0, y-1), y+2)
        print("[Agents]")
        print(self.agents[chopper])
        print("[env]")
        print(self.env[chopper])
        print("[Scores]")
        print(scores[chopper])
        print("[Surroundings]")
        print(int(surrounding[x,y]))
        print("[Best]")
        print(best[chopper])
        return self.agents, self.env, scores, surrounding, best

    def getTrustPair(self, ident1, ident2):
        return self.trust_network[ident1 - 1, ident2 - 1]

    def getTrustAverage(self, loc):
        x, y = loc
        total = 0
        num_adj = 0
        ident1 = self.locations[(x,y)]
        for i in [1,-1]:
            for j in [1, -1]:
                idx = (x + i, y + j)
                if idx in self.locations:
                    ident2 = self.locations[idx]
                    total += self.getTrustPair(ident1, ident2)
                    num_adj += 1
        if num_adj == 0:
            return .5
        return total/num_adj


    # private functions
    def _countScore(self, num):
        total = 0
        while num > 0:
            digit = num % 0x10
            total += self.rule[digit]
            num //= 0x10
        return total

    def _seed(self):
        np.random.seed(self.seed)
        self.seed += 1

    def _findBest(self, num):
        """
        Given number calculated by correlate2d,
        find the location (relative to self)
        where it has the highest score
        Encoded as middle = 0, top = 1, left = 2, right = 3, bottom = 4
        """
        middle, num = num % 100, num//100
        top, num = num % 100, num//100
        left, num = num % 100, num//100
        right, num = num % 100, num//100
        bottom = num % 100
        d = {0:middle, 1:top, 2:left, 3:right, 4:bottom}
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

