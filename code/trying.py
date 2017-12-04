import numpy as np
from scipy.signal import correlate2d
import matplotlib.pyplot as plt
from matplotlib import colors, animation
import random


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
    bounds=[-.5,.5,1.5,2.5,3.5,4.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    def __init__(self, L=49, seed=0, R=10, S=0, T=13, P=1, N=3, **config):
        self.size = size = (L, L)
        # Encode 'env' matrix to have
        # Empty location as 0, Defector as 1, Cooperator as 2
        self.env = np.zeros(size, dtype=np.uint64)

        # Register vectorized count score function and get state function
        self.vcountScore = np.vectorize(self._countScore)
        self.vgetState = np.vectorize(lambda x: x.state if x is not None else 0)
        self.vgetColor = np.vectorize(lambda x: x.color if x is not None else 0)
        self.vfindBest = np.vectorize(self._findBest)

        # Make a matrix that holds Agent objects
        self.agents = np.full(size, None)

        # Register seed
        self.seed = seed

        # Register R,S,T,P
        self.rule = {10:R, 9:S, 6:T, 5:P, 0:0, 1:0, 2:0, 4:0, 8:0}

        # Register configurations
        self.config = config

        # Migration parameter
        self.neighborhood = N

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

        for idx in all_indices[chosen_indices,:]:
            self._seed()
            initial_state = np.random.choice([1,2])
            self._seed()
            self.agents[tuple(idx)] = Agent(initial_state, seed=self.seed)

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
        self.scores = self.vcountScore(c) # n x n nparray
        #print(self.scores)
        # Find best for each cell
        surrounding = correlate2d(self.scores, self.kernel2, mode="same")
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
            agent.choose_next_state()


        if self.config.get("migrate"):
            self.migrate()

        self.update_env()

        return self.scores

    def migrate(self):
        """
        Randomly migrate to different empty location
        You can override this function by inheriting Environment class
        and implementing your own migrate(self) function
        """
        n, _ = self.size
        env = self.env
        agents = self.agents

        # Probability of relocation
        p = 0.05

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
        # for each migrating agent, find empty cells by N
        for source in migrator_indices:
            empty_cells_list = []
            source = tuple(source)
            for cell in empty_indices:
                cell = tuple(cell)
                if cell[0] - self.neighborhood <= source[0] <= cell[0] + self.neighborhood:
                    if cell[1] - self.neighborhood <= source[1] <= cell[1] + self.neighborhood:
                        empty_cells_list.append(cell)

            for cell in empty_cells_list:
                self.env[cell] = self.env[source]

            c = correlate2d(self.env, self.kernel, mode="same")
            self.scores = self.vcountScore(c) # n x n nparray

            scores_empty = dict()
            for cell in empty_cells_list:
                scores_empty[cell[0],cell[1]] = self.scores[cell[0]][cell[1]]
                self.env[cell] = 0

            max_score= max(scores_empty.values())
            max_cell = [k for k,v in scores_empty.items() if v == max_score]
            max_cell = max_cell[random.randrange(len(max_cell))]
            #print(len(np.argwhere(self.env > 0)))
            #move
            agents[max_cell] = agents[source]
            agents[source] = None
            for i in empty_indices:
                if i[0] == max_cell[0]:
                    if i[1] == max_cell[1]:
                        index = i
                        break

            empty_indices[index] = list(source)

            self._seed()

        return migrator_indices

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
        #print("[Agents]")
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
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import rc
    rc('animation', html='html5')
    env = Environment(migrate=True)
    env.place_agents()
    env.playRound()
    #env.debug(2,3)
    env.visualize()
    env.animate()
