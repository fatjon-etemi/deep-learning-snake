import random
import sys

import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

from Snake import Game
from plothistory import plot_training

LR = 1e-3
goal_steps = 500
games_to_evaluate = 5
only_plot_history = False
env = Game()
nn = True
balancing = True
balancing_factor = 4
selected_dataset = 'saved_expert_player.npy'
train_data_factor = 0.75


def create_neural_network_model():
    network = Sequential()
    network.add(Flatten(input_shape=(24, 1)))
    network.add(Dense(64, activation='relu'))
    network.add(Dropout(0.2))
    network.add(Dense(32, activation='relu'))
    network.add(Dense(3, activation='softmax'))
    network.compile(optimizer=Adam(lr=LR), loss='categorical_crossentropy', metrics=['accuracy'])

    return network


def train_model(training_data_input, input_model):
    shape_second_parameter = len(training_data_input[0][0])
    random.shuffle(training_data_input)

    train_data = training_data_input[:int(len(training_data_input) * train_data_factor)]
    x_train = np.array([i[0] for i in train_data])
    x_train = x_train.reshape(-1, shape_second_parameter)
    y_train = [i[1] for i in train_data]

    test_data = training_data_input[int(len(training_data_input) * train_data_factor):]
    x_test = np.array([i[0] for i in test_data])
    x_test = x_test.reshape(-1, shape_second_parameter)
    y_test = [i[1] for i in test_data]

    history = input_model.fit(x_train.tolist(), y_train, validation_data=(x_test.tolist(), y_test), epochs=10, batch_size=16)

    return input_model, history


def train_custom_model(training_data_input):
    shape_second_parameter = len(training_data_input[0][0])
    x = np.array([i[0] for i in training_data_input])
    x = x.reshape(-1, shape_second_parameter)
    y = np.array([np.argmax(i[1]) for i in training_data_input])
    clf = RandomForestClassifier()
    clf.fit(x, y)
    return clf


def evaluate(compiled_model):
    scores = []
    choices = []
    for each_game in range(games_to_evaluate):
        score = 0
        game_memory = []
        prev_obs = []
        env.reset()
        for _ in range(goal_steps):
            env.render()

            if len(prev_obs) == 0:
                action = random.randrange(0, 3)
            else:
                if nn:
                    prediction = compiled_model.predict(prev_obs.reshape(-1, len(prev_obs), 1))
                    action = np.argmax(prediction[0])
                else:
                    prediction = compiled_model.predict(prev_obs.reshape(1, -1))
                    action = prediction[0]

            choices.append(action)
            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            game_memory.append([new_observation, action])
            score += reward
            if done:
                break

        scores.append(score)
    print('Average Score:', sum(scores) / len(scores))
    return sum(scores) / len(scores)


if __name__ == '__main__':
    training_data = np.load(selected_dataset, allow_pickle=True)[0]
    # balancing the data
    if balancing:
        training_data_0 = [i for i in training_data if np.argmax(i[1]) == 0]
        training_data_1 = [i for i in training_data if np.argmax(i[1]) == 1]
        training_data_2 = [i for i in training_data if np.argmax(i[1]) == 2]
        training_data = training_data_0 + training_data_1 + random.choices(training_data_2, k=int(len(training_data_2) / balancing_factor))
    print("Training data length:", len(training_data))
    if nn:
        model = create_neural_network_model()
        model, history = train_model(training_data, model)
    else:
        model = train_custom_model(training_data)
    if only_plot_history:
        plot_training(history)
    else:
        evaluate(model)
    sys.exit()
