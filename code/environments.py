import numpy as np
from scipy.signal import correlate2d
import matplotlib.pyplot as plt
from matplotlib import colors, animation

from agents import Agent, SocialAgent

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

    # Count kernel is used to count defectors or cooperators
    count_kernel = np.array([[0, 1, 0],
                             [1, 0 ,1],
                             [0, 1 ,0]], dtype=np.uint64)

    # Things needed for plotting
    cmap = colors.ListedColormap(['white', 'red', 'blue', 'green', 'yellow'])
    bounds=[0,1,2,3,4]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    def __init__(self, L=49, seed=0, R=10, S=0, T=13, P=1, **config):
        """
        L : Length of one side
        seed : Random seed for controlling randomness
        T : Temptation payoff
        R : Reward for cooperating
        P : Punishment payoff for both defect
        S : Sucker's payoff
        (T > R > P > S) condition indicates prisoner's dillema condition
        (T > R > P > S) condition indicates snowdrift game
        config : possible configs - 'migrate'
        """
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
            agent.choose_next_state()


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
        print("[Colors]")
        print(self.vgetColor(self.agents)[chopper])
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


class SocialNetworkEnv(Environment):
    # Overriding place_agents function
    def place_agents(self):
        """
        Places agents randomly and leave 50% empty sites
        You can override this function by inheriting Environment class
        and implementing your own place_agents(self) function
        """
        L, _ = self.size
        all_indices = np.array(list(np.ndindex(self.size)))
        self._seed()
        number_of_agents = L*L//2
        chosen_indices = np.random.choice(L*L, number_of_agents, replace=False)

        self.agent_list = []
        # Initialize social trust network
        self.agent_network = np.random.random((number_of_agents, number_of_agents))
        for idnumber, idx in enumerate(all_indices[chosen_indices,:]):
            self._seed()
            initial_state = np.random.choice([1,2])
            self._seed()
            agent = SocialAgent(initial_state, seed=self.seed)
            agent.idnumber = idnumber
            self.agents[tuple(idx)] = agent
            self.agent_list.append(agent)
        self.update_env()

    def migrate(self, M=5):
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
        empty_indices_set = set(map(tuple, empty_indices))

        # Count Defector-Cooperator ratio
        #num_cooperator = correlate2d(self.env == 2, self.count_kernel, mode="same")
        #num_defector = correlate2d(self.env == 1, self.count_kernel, mode="same")
        #denominator = num_cooperator + num_defector
        #ratio = np.divide(num_cooperator, denominator, out=np.zeros_like(num_cooperator), where=(denominator!=0))

        # for each migrating agent, choose best destination
        for source in migrator_indices:
            source = tuple(source)

            # generate source agent's range
            x, y = source
            range_indices_set = set((x+i, y+j) for i in range(-M, M+1) for j in range(-M, M+1) if not (i==0 and j==0))

            # Get empty spots within the range (intersection)
            empty_in_range = range_indices_set & empty_indices_set

            # If no empty spots, don't migrate
            if len(empty_in_range) == 0:
                continue

            source_agent = agents[source]
            s_id = source_agent.idnumber

            # For each empty spots, calculate average trust value
            # And calculate the best empty spot that has the highest expected average trust value
            current_max = -1
            for d_x, d_y in empty_in_range:
                trust_value_sum = 0
                count = 0
                for a in [d_x-1, d_x+1]:
                    for b in [d_y-1, d_y+1]:
                        if not (0 <= a < n and 0 <= b < n):
                            continue
                        if agents[a, b] is not None:
                            d_id = agents[a, b].idnumber
                            trust_value_sum += self.agent_network[s_id, d_id]
                            count += 1
                if count == 0:
                    if current_max == -1:
                        dest = (d_x, d_y)
                    continue
                average_value = trust_value_sum / count
                if average_value > current_max:
                    current_max = average_value
                    dest = (d_x, d_y)

            # move
            agents[dest] = agents[source]
            agents[source] = None
            empty_indices_set.remove(dest)
            empty_indices_set.add(source)


    def playRound(self):
        """
        Simulates Prisoner's Dillema, calculates score for each cell, and updates agents
        """
        n, _ = self.size

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

            # Calculate the average trust values
            s_id = agent.idnumber
            trust_value_sum = 0
            count = 0
            for a in [x-1, x+1]:
                for b in [y-1, y+1]:
                    if not (0 <= a < n and 0 <= b < n):
                        continue
                    if self.agents[a, b] is not None:
                        d_id = self.agents[a, b].idnumber
                        trust_value_sum += self.agent_network[s_id, d_id]
                        count += 1
            if count == 0:
                average_value = 0.5
            else:
                average_value = trust_value_sum / count

            agent.update_average_neighbor_weights(average_value)
            agent.choose_next_state()


        if self.config.get("migrate"):
            self.migrate()

        self.update_env()

        return scores




