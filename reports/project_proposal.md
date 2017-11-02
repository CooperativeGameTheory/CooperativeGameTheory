# Modeling cooperation in the spatial PD with noise and biased migration
*Noah Rivkin, Katya Donovan, Young Seok Kim*

Following Helbing’s and Yu’s *The outbreak of cooperation among success-driven individuals under noisy conditions*, we will investigate the self-organization of cooperative behavior. By modeling unrelated individuals with no behavioral traits and no social networks who are success-driven, we examine when and why cooperation occurs, even when the model makes cooperation difficult.

We will simulate Prisoner’s Dilemma game on two dimensional spatial environment with 2 kinds of noises. First, we will add strategy mutation noise. With probability r, each individual will spontaneously choose to cooperate with probability q or to defect with probability 1-q until the next strategy change. Second, we will add random relocation noise. With probability r, an individual randomly chooses a free site without expected success. We will do all 4 combinations of experiments whether to add each of the noises.

To extend our experiments, we are hoping to add some of the following : biased relocation, various types of agents, social networks, and population pressure. Rather than randomly choosing a free site, we want the agent to choose a location where the likelihood of success is higher. Also, we want to introduces different types of agents and see how they interact. In addition, we are hoping to add social network to the experiments. Lastly, we want to see how agents behave given the population pressure. For example, resource or age restriction. 

We expect to see the migration of our agents across the grid, forming clusters that are cooperative after several time steps. A possible graph of our experiment follows.
![Figure 1](/figures/clusters.png "Figure 1")

In order to analyze our model, we will determine the number of cooperators and defectors present at each time step. The dominant strategy should be the more numerous of the two. In order to measure the stability of the configuration, we will keep track of how many agents change between cooperators and defectors for each time step. If we implement social groups or distinguishing characteristics for agents we will graph the changes in their populations over time. If we implement population pressure we will graph the average attributes of the agents and determine if some attributes are becoming more prevalent. If the average is increasing then we assume that the trait is beneficial under the conditions present at the time. Some traits may be beneficial only under certain conditions, such as the presence of noise over a certain threshold. We will control this possibility by sweeping parameters independently, isolating the conditions required.

When implementing our model, there are some areas for concern. Since it is difficult to split up the work into tasks that allow each member to work individually, integration and communication will be necessary to confirm that our code works together. Additionally, there are a plethora of factors that can be added to the model in order to represent the real world more accurately. However, knowing which factors will give meaningful results is more difficult. When we analyze the results of our model, we will have to be certain that we understand which of the model’s attributes affect the data in order to form accurate conclusions. 

Over the next week, each of us will be begin implementing certain aspects of the model. Tony will begin by setting up the environment for the model, which includes implementing the step function and thus choosing the number of agents that will move, the locations that they move to, etc. Katya will begin working on adding noise to the model, which will be implemented three different ways. The first method for noise is assuming that an individual spontaneously switches between cooperation and defection. The second method is individuals move randomly to an empty site without considering the surroundings of the empty space. The third implementation of noise is a combination of the first two, assuming that individuals randomly relocate and randomly change strategies. Finally, Noah will create the agents, which means setting up the class that creates agents. As part of an extension, Noah will work on personalizing the agents on a certain characteristic. Additionally, Noah will work on visualizing the model most effectively. 


## Bibliography
### The Further Evolution of Cooperation
*Axelrod, R., & Dion, D. (1988). The further evolution of cooperation. Science, 242(4884), 1385-1390.*

Axelrod and Douglas argue that evolution can produce cooperation as a trait, even in situations where defection is a dominant strategy in a non-iterated version of the situation. They expand on the prisoner's dilemma tournament model used in Axelrod’s original tournament to incorporate evolutionary aspects. In the new model automata are constructed from a set of instructions, and then they interact with other automata as is the case in the older model. However, in addition to altering the population of each type of automata, the automata exchange instructions (chromosomes) or randomly alter a single instruction (mutation). Axelrod and Douglas conclude that cooperation can emerge from randomized evolution.


### The outbreak of cooperation among success-driven individuals under noisy conditions 
*Helbing, D., & Yu,W. (2008).*

The outbreak of cooperation among success-driven individuals under noisy conditions. The National Academy of Sciences of the USA.
Helbing and Yu investigate the emergence of cooperative behavior among competitive agents when non-cooperative behavior is a dominant strategy. They use the prisoner’s dilemma as a model for such interactions. They make the argument that the common solution to the dilemma, iteration, is not able to account for changing situations, or deal with a noisy environment. They construct an agent based model to create a more realistic model, implementing changing strategies, noise, and changing spatial dynamics. They find that introducing migration aids in the emergence of cooperation, even though individual agents no longer interact with the same set of other agents repeatedly.


### The Evolution of Strategies in the Iterated Prisoner’s Dilemma 
*Axelrod, Robert. “The Evolution of Strategies in the Iterated Prisoner’s Dilemma.” The Dynamics of Norms, edited by Cristina Bicchieri and Richard Jeffrey.*

Axelrod explores different models to study the evolution of strategies in games. He mainly examines the strategies in the Prisoner’s Dilemma, where exploiting the cooperation of others gives high rewards, unless both are defective. By investigating which strategies will produce the highest reward, Axelrod shows that simple strategies such as choosing the strategy that the prisoner’s partner had recently tried will be an effective way of earning the highest reward.





