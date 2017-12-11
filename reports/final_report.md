# Observing Cooperation in Social Networks
## Katya Donovan, Tony Kim, Noah Rivkin
## Abstract

Following Helbing’s and Yu’s model on cooperation of success-driven individuals [1], we explore how noise affects the outbreak of cooperation among agents in the spatial prisoner’s dilemma. In Helbing’s and Yu’s model, they implement noise in two ways: one as strategy mutations in which an agent spontaneously chooses to cooperate or defect, and one where an agent chooses to migrate to another location, either randomly or to the best location within a Moore neighborhood, or the two dimensional square lattice with a central cell. With this model, we can see clusters of cooperation, even when the individuals are success driven. As an extension to this model, we investigate the effects of a trust based migration on communities of cooperators and defectors, where trust is a relation between pairs of agents based on past interactions. We analyze this with origin based migration, so that the agents who are unhappy, or surrounded by defecting cells, move to a random location. We also implement changes in strategies between agents based on their trust values.


## Introduction

The prisoner’s dilemma (PD) [2] is commonly investigated in game theory. Two agents are competing, and each one must choose one of two strategies, cooperate (C), or defect (D). If both agents cooperate they both receive a reward (R). Temptation (T), is an incentive to defect. If one agent defects and the other cooperates, the defector gets T while the cooperator gets the sucker (S) payoff. The prisoner’s dilemma is formally defined by T>R>P>S, and 2R>T+S. These rules mean that defection is what is known as a dominant strategy. A dominant strategy is a strategy that will always give a greater payoff than any other strategy, regardless of the other actions of the other agent or agents. However, the combined score of the two agents is greatest if they both cooperate, so from the perspective of maximizing the total payoff for a group of agents mutual cooperation is preferable. Unfortunately, for a single round of the PD, two rational agents will both choose D, giving the worst possible total payoff. A rational agent is an agent that will always act in its own best interests

There are a number of ways of resolving the PD. A common solution is the iterated prisoner’s dilemma, or IPD [2]. In the IPD the agents play each other repetitively. This means that establishing a situation where both agents always cooperate is preferable to a situation where both agents always defect. Another solution in the spatial PD. In the spatial PD agents are placed on a grid, or similar spatial structure, and play their neighbors repetitively. Like the IPD, it is in the agent's best interest to establish a pattern of cooperation.

Both of these solutions require repeated interaction between a pair of agents. In the real world most people interact with a group of people repeatedly. However, as communication and travel technologies grow more advanced, it is increasingly common to have single interactions with another person. This casts doubt on the traditional solutions to the prisoner’s dilemma.

We attempt to model the increased number of non-repeated interactions by implementing an altered version of the spatial PD[2]. In the model agents change their locations (migrate). This means that agents cannot assume that they will interact with a particular individual multiple times, reducing the incentive to establish a cooperative relationship with other agents. This allows us to test if cooperation will still emerge without the guarantee of repeated interactions.

Helbing and Yu created a model to study this situation, but their model lacks certain features that we consider necessary to create a model that represents a realistic environment. In their model agents are chosen to migrate at random and agents do not remember interactions with specific individuals. We implement a trust network between agents, providing agents with an incentive to migrate away from individuals with whom they have had negative past interactions.This models social groups, where individuals try to find agents they consider reliable, and attempt to maintain long term relations with them.



## Methodology

Helbing and Yu implement a spatial prisoner's dilemma with a variety of initial conditions in order to analyze the effects of noise on the formation of communities of cooperators and defectors. They examine the effect of having a percentage of the agents imitate the strategy of the neighbor that was most successful after each round and the effect of having agents migrate to destinations where they have the best chance of being successful.  The imitation of successful strategies and the success driven migration are both treated as a type of noise This results in agents migrating to a nearby cluster of cooperators. We compare the results of implementing these types of noise both separately as well as simultaneously in order to observe whether communities of cooperators will arise. We add strategy mutation noise and migration, and observe the stability of the emergence of cooperation in the spatial prisoner’s dilemma. This allows us to determine the robustness of the model. By combining these implementations in four different ways, we analyze the impact of these factors on the outbreak of clusters of cooperators or defectors.

We extend Helbing and Yu’s model by implementing social dynamics. We implement trust relationships by giving each possible pair of agents a trust value. This pairwise trust network is used by the agents to decide on strategies. We also make migration depend on this trust network. We analyze the strategy and spatial changes within the simulation. 

The trust relations between agents are initialized with a random trust value that is between 0 and 1. This represents the level of trust between the agents. Each round the agents take the average value of their trust with each adjacent agent, and choose to cooperate if the average trust value (ATV) is over a threshold of .75. If an agent defects their trust values with the agents they interacted with decreases, and if they cooperate the trust value increases.

In order to incorporate the trust network into the model, we need to make migration dependent on the trust network we built. Agents are chosen to migrate based on their ATV (origin based migration).  We select a percentage, p, of the agents with the lowest ATV to migrate. We typically choose p=0.05. The migrating agents move to a random destination. We believe this will model the behavior of unhappy people who wish to migrate from a neighborhood of defectors.

## Results
Starting with a 49 x 49 grid with 50% of empty cells and an approximately equal amount of defectors and cooperators, we examine the clustering of cooperators and defectors after 200 iterations. In order to verify our results, we compare our implementations against those of the models, starting with the simplest version of the model, and adding more factors such as noise and migration as the model becomes more complicated.

We first consider the model that does not implement noise or migration. The only remaining way that agents change state is by imitating their highest scoring neighbor. Figure 1 shows Helbing’s and Yu’s model in contrast to Figure 2, which depicts our own implementation of this model. As clearly indicated by the figure, after 200 iterations, our model has less cooperators than Helbing’s and Yu’s. This suggests that Helbing and Yu use a different method to calculate scores than we implemented in our model. This may be due bias in pseudo random number generators, or in our treatment of ties between highest scoring neighbors. However, since the defectors remain a majority in both implementations, we consider our model to be a qualitatively accurate replication of theirs.

|![Figure 1](/figures/A_model.jpeg "Figure 1: Helbing and Yu’s model without noise after 200 timesteps")| ![Figure 2](/figures/A.png "Figure 2: Our implementation of Helbing and Yu’s model after 200 timesteps")|
|:----------:|:----------:|
| Figure 1: Helbing and Yu’s model without noise after 200 timesteps | Figure 2: Our implementation of Helbing and Yu’s model after 200 timesteps| 


We add noise to our model in the form of random strategy mutations. In this implementation cooperators and defectors have a 5% chance of switching to a random state each timestep. This noise eliminates the majority of the cooperators in the model, as can be seen Figure 3, which is Helbing’s and Yu’s model, and Figure 4, our own implementation. Since cooperators are mostly eliminated, we observe that imitation based strategies is not robust against random mutations. This is because cooperative cluster can easily be invaded when a member of the cluster mutates into a defector. At that point the defector will be completely surrounded by cooperators. It will have a score of 4T, the maximum possible score for a single agent, and its neighbors will imitate it. This can eliminate the cooperative cluster in only a few rounds.

|![Figure 3](/figures/D_model.jpeg "Figure 3") "Figure 3: Helbing and Yu’s model with random strategy mutation after 200 timesteps")| ![Figure 4](/figures/D.png "Figure 4")|
|:----------:|:----------:|
| Figure 3: Helbing and Yu’s model with random strategy mutation after 200 timesteps| Figure 4: Our model with random strategy mutations after 200 timesteps|


Figure 5 depicts Helbing’s and Yu’s implementation of the model without random strategy mutation, instead implementing migration. Figure 6 is our own implementation of the model with those conditions. As shown in the two figures, small clusters of cooperation appear, because random agents will chose to migrate to a place where their score is higher, which is overall higher for a collective group of cooperators than for a group of defectors, indicating why clusters may appear.


|![Figure 5](/figures/B_model.jpeg "Figure 5") "Figure 3: Helbing and Yu’s model with random strategy mutation after 200 timesteps")| ![Figure 6](/figures/B.png "Figure 6")|
|:----------:|:----------:|
| Figure 5:  Helbing’s and Yu’s model with migration after 200 timesteps|Figure 6: Our model with migration after 200 timesteps|

However, the clusters of cooperation are unstable, because once we add noise to the model, as in random chance that the strategy of the agents change, the clusters of cooperation cease. One defector can eliminate a cluster of cooperators, by taking advantage of their strategy, which turns them into defectors the next time the model is run. Figure 7, Helbing’s and Yu’s model, and Figure 8, our own implementation of the model, clearly show this elimination.

|![Figure 7](/figures/E_model.jpeg "Figure 7") "Figure 3: Helbing and Yu’s model with random strategy mutation after 200 timesteps")|![Figure 8](/figures/E.png "Figure 8")|
|:----------:|:----------:|
| Figure 7: Helbing and Yu’s implementation of the model with destination based migration, in addition to random migration noise after 200 timesteps|Figure 8: Our model with destination based migration, in addition to random migration noise after 200 timesteps|

We extend our model by implementing a dynamic social network to our agents, which affects the agent’s strategies and migration. Agents now follow an origin based migration where agents with low trust values move to a random location. We believe that by implementing a social network, we will observe more clusters of cooperators after a shorter time frame. Figure 9 shows the results from this implementation after 200 time step iterations. As predicted, there is a higher amount of clustering than without a social network. There are also more cooperators, which indicates that trust can build up among agents fairly quickly.  

|![Figure 9](/figures/extend.png "Figure 9")
|:----------:|
| Figure 9: Our extension of the model with trust based strategy and migration after 200 timesteps|

We use our extension to simulate relationships among people in the real world, as humans are inherently biased towards people who have been kind to them in the past. By implementing trust based strategies and migration, we can mimic our own bias towards people who have been either trustworthy or untrustworthy. With this imitation, we find small communities of cooperators, similar to how societies are structured in the real world.

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
