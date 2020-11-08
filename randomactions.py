import random
from Snake import Game
from collections import Counter
from statistics import median, mean
import os
import numpy as np

LR = 1e-3
goal_steps = 100
score_requirement = 100
initial_games = 10000
env = Game()
nn = True


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
            if os.path.exists('./saved_random_actions_model' + str(score_requirement) + '.npy'):
                prev_training_data = np.load('pred_save' + str(score_requirement) + '.npy', allow_pickle=True)[0]
                my_training_data = my_training_data + prev_training_data
            training_data_save = np.array([my_training_data, score_requirement])
            np.save('saved_random_actions_model' + str(score_requirement) + '.npy', training_data_save)
        else:
            training_data_save = np.array([my_training_data, score_requirement])
            np.save('saved_random_actions.npy', training_data_save)

    return my_training_data


if __name__ == '__main__':
    generate_population(None)
