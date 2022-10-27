from easy21Env import *
import numpy as np

def MonteCarlo(value, counter, N_zero) :
	env = Easy21Env()

	trace = []
	reward = 0
	while not env.terminate :
		e = N_zero / (N_zero + np.sum(counter[:, env.player.points, env.dealer.points], axis = 0))
		if np.random.random() < e :
			action = np.random.randint(0, 2)
		else :
			action = np.argmax(value[:, env.player.points, env.dealer.points])
		trace.append((action, env.player.points, env.dealer.points))
		counter[action, env.player.points, env.dealer.points] += 1
		temp = env.step(action)
		reward += temp

	for (action, player_points, dealer_points) in trace :
		value[action, player_points, dealer_points] += 1 / counter[action, player_points, dealer_points] * (reward - value[action, player_points, dealer_points])

	return value, counter







