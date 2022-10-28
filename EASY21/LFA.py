from easy21Env import *
import numpy as np 

def feature_mapping(action, player_points, dealer_points) :
	feature = np.zeros(3 * 6 * 2)
	for i, (lower, upper) in enumerate(zip(range(1, 8, 3), range(4, 11, 3))) :
		if lower <= dealer_points <= upper:
			feature[i] = 1
	for i, (lower, upper) in enumerate(zip(range(1, 17, 3), range(6, 22, 3)), start = 3) : 
		if lower <= player_points <= upper :
			feature[i] = 1
	if action == 1 :
		feature[-1] = 1
	else : feature[-2] = 1

	return feature

def get_value_approximation(action, player_points, dealer_points, weight) :
	return np.dot(feature_mapping(action, player_points, dealer_points), weight)

def epsilon_greedy(weight, player_points, dealer_points, epsilon) :
	if np.random.rand() >= epsilon :
		if get_value_approximation(0, player_points, dealer_points, weight) <= get_value_approximation(1, player_points, dealer_points, weight) :
			action = 1
		else : action = 0
	else : action = np.random.randint(0, 2)

	return action

def SARSA(weight, lamb, epsilon, step_size) :
	elgibility = np.zeros(3 * 6 * 2)
	env = Easy21Env()

	action = epsilon_greedy(weight, env.player.points, env.dealer.points, epsilon)
	state = (env.player.points, env.dealer.points) 
	while not env.terminate:
		reward = env.step(action)
		if not env.terminate:
			new_action = epsilon_greedy(weight, env.player.points, env.dealer.points, epsilon)
			delta = reward + get_value_approximation(new_action, env.player.points, env.dealer.points, weight) - get_value_approximation(action, state[0], state[1], weight)
		else :
			delta = reward - get_value_approximation(action, state[0], state[1], weight)
			new_action = action

		elgibility = elgibility * lamb + feature_mapping(action, state[0], state[1])
		weight = weight + step_size * delta * elgibility

		action = new_action
		state = (env.player.points, env.dealer.points)

	return weight
