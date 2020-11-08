import random
import sys
from collections import Counter
from statistics import median, mean
import os.path
#import matplotlib.pyplot as plt

import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

from Snake import Game

LR = 1e-3
goal_steps = 500
score_requirement = 500
initial_games = 10
env = Game()


def generate_population(model_input):
    # [OBS, MOVES]
    global score_requirement

    my_training_data = []
    # all scores:
    scores = []
    # just the scores that met our threshold:
    accepted_scores = []
    # iterate through however many games we want:
    print('Score Requirement:', score_requirement)
    for _ in range(initial_games):
        # env = game()
        print('Simulation ', _, " out of ", str(initial_games), '\r', end='')
        # reset env to play again
        env.reset()

        score = 0
        # moves specifically from this environment:
        game_memory = []
        # previous observation that we saw
        prev_observation = []
        choices = []
        # for each frame in 200
        for _ in range(goal_steps):
            # choose random action (0 or 1)
            if len(prev_observation) == 0:
                action = random.randrange(0, 3)
            else:
                if not model_input:
                    action = random.randrange(0, 3)
                else:
                    if nn:
                        prediction = model_input.predict(prev_observation.reshape(-1, len(prev_observation), 1))
                        action = np.argmax(prediction[0])
                    else:
                        prediction = model_input.predict(prev_observation.reshape(1, -1))
                        action = prediction[0]

            # do it!
            choices.append(action)
            repeater_length = random.randrange(1, 20) * -1
            if len(choices) > repeater_length * 2 and choices[repeater_length:] == choices[
                                                                                   repeater_length * 2:repeater_length] and choices[
                                                                                                                            repeater_length:0] != [
                2] * (repeater_length * -1):
                action = random.randrange(0, 3)
            observation, reward, done, info = env.step(action)
            # notice that the observation is returned FROM the action
            # so we'll store the previous observation here, pairing
            # the prev observation to the action we'll take.
            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score += reward
            if done:
                break

        # IF our score is higher than our threshold, we'd like to save
        # every move we made
        # NOTE the reinforcement methodology here.
        # all we're doing is reinforcing the score, we're not trying
        # to influence the machine in any way as to HOW that score is
        # reached.
        if score > score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                # convert to one-hot (this is the output layer for our neural network)

                action_sample = [0, 0, 0]
                action_sample[data[1]] = 1
                output = action_sample
                # saving our training data
                my_training_data.append([data[0], output])

        # save overall scores
        scores.append(score)

    # some stats here, to further illustrate the neural network magic!
    if len(accepted_scores) > 0:
        print('Average accepted score:', mean(accepted_scores))
        print('Score Requirement:', score_requirement)
        print('Median score for accepted scores:', median(accepted_scores))
        print(Counter(accepted_scores))
        # score_requirement = mean(accepted_scores)

        # just in case you wanted to reference later
        if model_input:
            if os.path.exists('./pred_save' + str(score_requirement) + '.npy'):
                prev_training_data = np.load('pred_save' + str(score_requirement) + '.npy', allow_pickle=True)[0]
                my_training_data = my_training_data + prev_training_data
            training_data_save = np.array([my_training_data, score_requirement])
            np.save('pred_save' + str(score_requirement) + '.npy', training_data_save)
        else:
            training_data_save = np.array([my_training_data, score_requirement])
            np.save('saved3.npy', training_data_save)

    return my_training_data


# def plotTraining(history):
#     # summarize history for accuracy
#     plt.figure(figsize=(14,6))
#     plt.subplot(1,2,1)
#     plt.plot(history.history['accuracy'])
#     plt.plot(history.history['val_accuracy'])
#     plt.title('model accuracy')
#     plt.ylabel('accuracy')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'valid'], loc='lower right')
#     plt.subplot(1,2,2)
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('model loss')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'valid'], loc='upper right')
#     plt.show()


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

    train_data_factor = 0.75

    train_data = training_data_input[:int(len(training_data_input) * train_data_factor)]
    x_train = np.array([i[0] for i in train_data])
    x_train = x_train.reshape(-1, shape_second_parameter)
    y_train = [i[1] for i in train_data]

    test_data = training_data_input[int(len(training_data_input) * train_data_factor):]
    x_test = np.array([i[0] for i in test_data])
    x_test = x_test.reshape(-1, shape_second_parameter)
    y_test = [i[1] for i in test_data]

    history = input_model.fit(x_train.tolist(), y_train, validation_data=(x_test.tolist(), y_test), epochs=10, batch_size=16)
    # plotTraining(history)
    # model.save('my_model')

    return input_model


def train_custom_model(training_data_input):
    shape_second_parameter = len(training_data_input[0][0])
    first_data = training_data_input[0][0]
    x = np.array([i[0] for i in training_data_input])
    x = x.reshape(-1, shape_second_parameter)
    y = np.array([np.argmax(i[1]) for i in training_data_input])
    clf = RandomForestClassifier()
    clf.fit(x, y)
    return clf


def evaluate(compiled_model):
    # now it's time to evaluate the trained model
    scores = []
    choices = []
    for each_game in range(10):
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
            #repeater_length = random.randrange(1, 20) * -1
            #if len(choices) > repeater_length * 2 and choices[repeater_length:] == choices[repeater_length * 2:repeater_length] and choices[repeater_length:0] != [2] * (repeater_length * -1):
            #    action = random.randrange(0, 3)
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
    nn = True

    training_data = np.load('saved_expert_player.npy', allow_pickle=True)[0]
    print(len(training_data))
    # balancing the data
    training_data_0 = [i for i in training_data if np.argmax(i[1]) == 0]
    training_data_1 = [i for i in training_data if np.argmax(i[1]) == 1]
    training_data_2 = [i for i in training_data if np.argmax(i[1]) == 2]
    training_data = training_data_0 + training_data_1 + random.choices(training_data_2, k=int(len(training_data_2) / 4))
    print("Training data length:", len(training_data))
    # training_data = generate_population(None)
    if nn:
        model = create_neural_network_model()
        model = train_model(training_data, model)
    else:
        model = train_custom_model(training_data)
    evaluate(model)
    #generate_population(model)
    sys.exit()
