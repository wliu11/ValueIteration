# ValueIteration

We implemented two dynamic programming algorithms, value prediction and value iteration.

In value prediction (also known as iterative policy evaluation), we iteratively sweep through all the states, updating the value function for each state using the Bellman optimality equation. Using the converged value functions, we can use the Bellman optimality equation again in order to compute the Q value for each state-action pair. 

In value iteration, we also sweep through all the states while updating the value function using the Bellman optimality equation. However instead of computing the Q-values, we calculate the optimal action to take from each state and use this to create the optimal policy. 

