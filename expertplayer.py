from Snake import Game as game
import pygame
from pygame.locals import *
from collections import Counter
from statistics import median, mean
import numpy as np
import os.path

LR = 1e-3
score_requirement = 1000
games_i_want_to_play = 3


def play_alone():
    my_training_data = []
    # all scores:
    scores = []
    # just the scores that met our threshold:
    accepted_scores = []
    # iterate through however many games we want:
    for _ in range(games_i_want_to_play):
        env = game()
    # reset env to play again
        env.reset()
        action = 2
        score = 0
    # moves specifically from this environment:
        game_memory = []
    # previous observation that we saw
        prev_observation = []
    # for each frame in 200
        while True:
            for event in pygame.event.get():
                if event.type == KEYDOWN:
                    if event.key == K_UP or event.key == K_w:
                        if env.snake.movedir == 'right':
                            action = 0
                        if env.snake.movedir == 'left':
                            action = 1
                        if env.snake.movedir == 'up':
                            action = 2
                        if env.snake.movedir == 'down':
                            action = 2
                    if event.key == K_DOWN or event.key == K_s:
                        if env.snake.movedir == 'right':
                            action = 1
                        if env.snake.movedir == 'left':
                            action = 0
                        if env.snake.movedir == 'up':
                            action = 2
                        if env.snake.movedir == 'down':
                            action = 2
                    if event.key == K_LEFT or event.key == K_a:
                        if env.snake.movedir == 'right':
                            action = 2
                        if env.snake.movedir == 'left':
                            action = 2
                        if env.snake.movedir == 'up':
                            action = 0
                        if env.snake.movedir == 'down':
                            action = 1
                    if event.key == K_RIGHT or event.key == K_d:
                        if env.snake.movedir == 'right':
                            action = 2
                        if env.snake.movedir == 'left':
                            action = 2
                        if env.snake.movedir == 'up':
                            action = 1
                        if env.snake.movedir == 'down':
                            action = 0
            # do it!
            env.render()
            observation, reward, done, info = env.step(action)
            # notice that the observation is returned FROM the action
            # so we'll store the previous observation here, pairing
            # the prev observation to the action we'll take.
            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score += reward
            action = 2
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
    if os.path.exists('./saved_expert_player.npy'):
        prev_training_data = np.load('saved_expert_player.npy', allow_pickle=True)[0]
        my_training_data = my_training_data + prev_training_data
    training_data_save = np.array([my_training_data, score_requirement])
    print('New training data length: ', len(training_data_save[0]))
    np.save('saved_expert_player.npy', training_data_save)

    return my_training_data


if __name__ == '__main__':
    play_alone()

