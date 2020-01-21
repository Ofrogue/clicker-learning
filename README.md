# clicker-learning
Improving an RL agent with a human teacher 
https://medium.com/@kmedyanovskiy/how-to-train-your-artificial-dragon-or-clicker-learning-778d24c6ead2

# Description

Humans train dogs by delivering rewards to specific actions so that the dog will associate certain actions and situations to either positive (to be repeated again) or negative (to be avoided) values. Similarly, artificial agents are now trained automatically by reinforcement signals delivered after a goal has been achieved by the agent. Unfortunately, this process is very slow and requires many samples to learn. The aim of this project is to test whether it is possible to train a character in a video-game by delivering the reward (mouse click) at any event of teacher's choice (not only at the end of a goal). How does it compare to state-of-the-art reinforcement learning algorithms?

# Methods
We are using:
* DQN algorithm
* gym emvironment
* stable baselines https://github.com/hill-a/stable-baselines
* with pretrained models from zoo https://github.com/araffin/rl-baselines-zoo
* gym enviroment https://github.com/openai/gym
* atari game Breakout

To conduct the experiment:
* adopt a state-of-the-art model from the zoo
* freeze all convolutional layers (or freeze all layers except the last $n$)
* add noise to the remained layers
* develop a customized reward mechanism based on a human reaction
  + run and render the enviroment
  + recieve a reward from user's clicks
  + update the unfrozen layers according to the reward
* run experiments with a human teacher

# Requirements & Installation
We suggest using conda  
```conda create -n clickerlearning```  
```conda activate clickerlearning```  

```conda install python==3.7 --yes && conda install -c conda-forge tensorflow --yes && conda install opencv --yes && pip install gym==0.11.0 stable-baselines==2.8.0 pygame && pip install gym[atari] ```  
stable-baselines are slightly outdated according to the latest changes in gym. Thus, we use an older version of gym.

# Application Manual
### Starting application
Navigate to ```App``` folder and run the following command ```python App.py``` and wait for the game to load.

After the game is loaded you will be able to interact with it, using the following keys:
* p - pause/unpause the game
* space - give reward
* q - hard reset(load the new model and environment)
* Esc - quit game

Your task is to press space each time you see the model does something good for your opinion. For examlpe, when it bounces the ball or destroys a brick.

At the top of the screen you will see 2 indicators: the number of iteraions left and 
the number of rewards that were given throughout the session.

After a session has ended(quit game, hard reset or the game has ended) you will see a new directory in ```App/tensor``` where both the model and 
some other meta data will appear.
