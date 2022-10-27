from easy21Env import *
import numpy as np 

def SARSA(value, counter, alpha, N_zero) :

	eligibility = np.zeros((2, 22, 11))
	env = Easy21Env()

	e = N_zero / (N_zero + np.sum(counter[:, env.player.points, env.dealer.points]))
	state = (env.player.points, env.dealer.points)
	if np.random.random() < e :
		action = np.random.randint(0, 2)
	else :
		action = np.argmax(value[:, env.player.points, env.dealer.points])

	while not env.terminate :

		counter[action, state[0], state[1]] += 1

		reward = env.step(action)
		new_action = action
		if not env.terminate :
			e = N_zero / (N_zero + np.sum(counter[:, env.player.points, env.dealer.points], axis = 0))
			if np.random.random() < e :
				new_action = np.random.randint(0, 2)
			else :
				new_action = np.argmax(value[:, env.player.points, env.dealer.points])
			delta = reward + value[new_action, env.player.points, env.dealer.points] - value[action, state[0], state[1]]
		else :
			delta = reward - value[action, state[0], state[1]]
		eligibility[action, state[0], state[1]] += 1
		step_size = 1 / (counter[action, state[0], state[1]])

		value +=  step_size * delta * eligibility
		eligibility *= alpha

		state = (env.player.points, env.dealer.points)
		action = new_action

	return value, counter


