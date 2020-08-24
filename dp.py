from typing import Tuple
import numpy as np
from env import EnvWithModel
from policy import Policy


'''
inp:
    env: environment with model information, i.e. you know transition dynamics and reward function
    pi: policy
    initV: initial V(s); numpy array shape of [nS,]
    theta: exit criteria
return:
    V: $v_\pi$ function; numpy array shape of [nS]
    Q: $q_\pi$ function; numpy array shape of [nS,nA]
'''
def value_prediction(env:EnvWithModel, pi:Policy, initV:np.array, theta:float) -> Tuple[np.array,np.array]:

    nA = env.spec.nA        # Number of actions
    nS = env.spec.nS        # Number of states
    
    V = np.zeros(nS)

    R = env.R               # Reward function
    T = env.TD              # State transition function
    gamma = env.spec.gamma  # Gamma

    """
    Iteratively sweep through all states, 
    """
    while True:
        delta = 0
        for s in range(nS):
            value = 0
            for a in range(nA):
                prob = pi.action_prob(s, a)
                for sp in range(nS):
                    value += (prob * (T[s, a, sp] * (R[s, a, sp] + gamma * initV[sp])))
                V[s] = value
            delta = max(delta, abs(initV[s] - V[s]))
        
        """ Check for convergence """
        if delta < theta:
            break 
        
        initV = V.copy()        # Must make an explicit copy
    
    """ 
    With the values function having converged, we can now extract
    the optimal action probabilities for each state, using the 
    Bellman optimality equation.
    """
    Q = np.zeros(shape=(nS, nA))
    for s in range(nS):
        for a in range(nA):
            for sp in range(nS): 
                Q[s, a] += T[s,a,sp] * (R[s,a,sp] + gamma * V[sp])
    
    return V, Q


        

'''
inp:
    env: environment with model information, i.e. you know transition dynamics and reward function
    initV: initial V(s); numpy array shape of [nS,]
    theta: exit criteria
return:
    value: optimal value function; numpy array shape of [nS]
    policy: optimal deterministic policy; instance of Policy class
'''

def value_iteration(env:EnvWithModel, initV:np.array, theta:float) -> Tuple[np.array,Policy]:
    
    nA = env.spec.nA        # Number of actions
    nS = env.spec.nS        # Number of states
    
    V = initV.copy()        # Value function
    
    pi = np.zeros(nS)       # Optimal policy
    R = env.R               # Reward function
    T = env.TD              # State transition function
    gamma = env.spec.gamma  # Gamma

    """ 
    Iteratively sweep through all states, actions, and resulting states,
    updating the expected value of each action for each state.
    We then record the optimal action for each state to form our optimal policy.
    """
    while True:
        delta = 0
        for s in range(nS):     
            action_values = []
            for a in range(nA):
                state_value = 0
                for sp in range(len(T[s][a])):
                    state_action_value = T[s,a,sp] * (R[s, a, sp] + gamma * initV[sp])
                    state_value += state_action_value
                action_values.append(state_value)
                best_action = np.argmax(np.asarray(action_values))
                
                """ Record the optimal action and the action value """
                V[s] = action_values[best_action]            
                pi[s] = best_action     
            
            """ Update delta and then check for convergence """
            delta = max(delta, abs(initV[s] - V[s]))
        if delta < theta:
            break
        else:
            initV = V.copy()        # Must make an explicit copy
            
    policy = OptimalPolicy(pi)
    return V, policy
 

class OptimalPolicy(Policy):
    def __init__(self, policy:np.array):
        self.policy = policy
      
    def action_prob(self,state:int,action:int) -> float:
        if self.policy[state] == action:
            return 1
        else: 
            return 0


    def action(self,state:int) -> int:
        return self.policy[state]
      
        
