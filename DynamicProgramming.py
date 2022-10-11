import numpy as np;


"""
P : nested dictionary, the transistion model
 P[s][a] = (Pr, next_s, r, terminate)
 	Pr : Probability that from state s move to state next_s with action a (float)
 	next_s: the state that agent move to (int)
 	r : reward of the transition (float)
 	terminate : definded whether the state is terminal (bool)

nS : number of states
nA : number of actions
policy : policy that agent must follow
	policy[s] : action that agent must do when in the state s
"""

def policy_evaluation(P, nS, nA, policy, gamma, tol = 1e-3):

	V = np.zeros(nS)
	while(True):
		temp_V = V.copy()
		V = np.zeros(nS)
		for s in range(nS):
			for (Pr, next_s, r, terminate) in P[s][policy[s]]:
				V[s] += Pr * (r + temp_V[next_s] * gamma)
		if(np.max(np.abs(V - temp_V)) < tol):
			break
	return V

def policy_improvement(P, nS, nA, policy_value, policy, gamma, tol = 1e-3):
	for s in range(nS):
		max_value = 0
		for a in range(nA):
			value = 0
			for (Pr, next_s, r, terminal) in P[s][a]:
				value += Pr * (r + policy_value[next_s] * gamma)
			if value > max_value:
				max_value = value
				policy[s] = a
	return policy

def policy_iteration(P, nS, nA, gamma, tol = 1e-3):

	old_policy = np.ones(nS, dtype = int)
	policy = np.zeros(nS, dtype = int)

	while np.any(old_policy != policy):
		old_policy = policy.copy()
		policy_value = policy_evaluation(P, nS, nA, policy, gamma, tol)
		policy = policy_improvement(P, nS, nA, policy_value, policy, gamma, tol)
	return policy

def value_iteration(P, nS, nA, gamma, tol = 1e-3):

	V = np.zeros(nS)

	while(True):
		old_V = V.copy();
		for s in range(nS):
			value = 0
			for a in range(nA):
				for(Pr, next_s, r, terminal) in P[s][a]:
					value += Pr * (r + old_V[next_s] * gamma)
				V[s] = max(value, V[s])
		if np.max(np.abs(V - old_V)) < tol: 
			break 

	policy = np.zeros(nS, dtype = int)
	policy = policy_improvement(P, nS, nA, V, policy, gamma, tol)
	return policy
