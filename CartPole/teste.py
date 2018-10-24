#imports
import gym
import random
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam


#variable initializing
env = gym.make('CartPole-v1')
env.reset()
goal_steps = 500
score_requirement = 60


def model_data_preparation(initial_games = 1000):
	training_data = []
	accepted_scores = []

	for game_index in range(initial_games):
		score = 0
		game_memory = []
		previous_observation = []

		for step_index in range(goal_steps):
			env.render()
			action = random.randrange(0, 2)
			observation, reward, done, info = env.step(action)

			if len(previous_observation) > 0:
				game_memory.append([previous_observation, action])

			previous_observation = observation
			score += reward

			if done:
				break

		if score >= score_requirement:
			accepted_scores.append(score)

			for data in game_memory:
				if data[1] == 1:
					output = [0, 1]
				elif data[1] == 0:
					output = [1, 0]

				training_data.append([data[0], output])

		print(game_index, ': ', score)

		env.reset()

	print(accepted_scores)

	return training_data


def build_model(input_size, output_size):
	model = Sequential()
	model.add(Dense(128, input_dim=input_size, activation='relu'))
	model.add(Dense(52, activation='relu'))
	model.add(Dense(output_size, activation='linear'))
	model.compile(loss='mse', optimizer=Adam())

	return model


def train_model(training_data):
	X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]))
	y = np.array([i[1] for i in training_data]).reshape(-1, len(training_data[0][1]))

	model = build_model(input_size=len(X[0]), output_size=len(y[0]))

	model.fit(X, y, epochs=10)

	return model


def play(model_file_name='model_1000.h5'):
	model = load_model(model_file_name)

	scores = []
	choises = []

	print('{}:'.format(model_file_name))

	for each_game in range(10):
		score = 0
		prev_obs = []

		for step_index in range(goal_steps):
			env.render()

			if len(prev_obs) == 0:
				action = random.randrange(0,2)
			else:
				action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs)))[0])

			choises.append(action)
			new_observation, reward, done, info = env.step(action)
			prev_obs = new_observation
			score += reward

			if done:
				break

		print(each_game+1, ': ', score)

		env.reset()
		scores.append(score)

	print('Average score: {}\n'.format(sum(scores)/len(scores)))


def generate_model(initial_games=1000):
	print('Training model with {} iterations'.format(initial_games))

	training_data = model_data_preparation(initial_games)
	trained_model = train_model(training_data)

	trained_model.save('model_{}.h5'.format(initial_games))	


if __name__ == '__main__':
	generate_model(initial_games=1600)
	play('model_1600.h5')
