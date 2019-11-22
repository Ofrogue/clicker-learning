# clicker-learning
Improving an RL agent with a human teacher 
# Description

Humans train dogs by delivering rewards to specific actions so that the dog will associate certain actions and situations to either positive (to be repeated again) or negative (to be avoided) values. Similarly, artificial agents are now trained automatically by reinforcement signals delivered after a goal has been achieved by the agent. Unfortunately, this process is very slow and requires many samples to learn. The aim of this project is to test whether it is possible to train a character in a video-game by delivering the reward (mouse click) at any event of teacher's choice (not only at the end of a goal). What would the best strategy for delivering a limited number of rewards (reward-shaping)? How does it compare to state-of-the-art reinforcement learning algorithms?

# Methods
We are using:
* DQN algorithm
* gym emviroment
* stable baselines https://github.com/hill-a/stable-baselines
* with pretrained models from zoo https://github.com/araffin/rl-baselines-zoo
* gym enviroment https://github.com/openai/gym
* atari game Breakout

To conduct the experiment we will:
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

```conda install python==3.7 --yes && conda install -c conda-forge tensorflow --yes && conda install opencv --yes && pip install gym==0.11.0 stable-baselines```  
stable-baselines are slightly outdated according to the latest changes in gym. Thus we use an older version of gym.
