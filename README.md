# deep-learning-snake

start Player.py to train the model and see some evaluation
  - Snake.py has to be in the same directory
  - some dataset has to be in the same directory (saved_random_actions.npy or saved_expert_player.npy)
  - set the selected_dataset variable
    - for saved_random_actions.npy it is recommended to set the balancing variable to False
    - for saved_expert_player.npy it is recommended to set the balancing variable to True and the balancing_factor to 4
    
 ## creating dataset
 
 With randomly generated actions: start randomactions.py
 Play snake yourself: start expertplayer.py
  - set the score_requirement variable (higher score requirement leads to higher performance after training the model)
  - set the games_i_want_to_play variable
  - if saved_expert_player.npy already exists, the data will be appended to that file

