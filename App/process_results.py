import os
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_scalars(dpath):
    summary_iterators = [EventAccumulator(os.path.join(dpath, dname)).Reload() for dname in os.listdir(dpath)]
    
    ea = None
    for ea in summary_iterators:
        if ea.Tags()["scalars"]:
            break
    table = [list(scalar_event) for scalar_event in ea.Scalars("episode_reward")]

    pd.DataFrame(table, columns=["wall_time", "step", "value"])
    return pd.DataFrame(table, columns=["wall_time", "step", "value"])


def get_rewards(dpath):
    path = os.path.join(dpath, "results.csv")
    blocks_data = pd.read_csv(path)
    blocks = blocks_data["blocks"].values
    blocks_per_frame = blocks[1:] - blocks[:-1]

    rewards_data = get_scalars(dpath)
    episode_ending_frames = rewards_data["step"]
    blocks_per_episode = []
    episode_begin = 0
    for episode_end in episode_ending_frames:
        episode_blocks = (blocks_per_frame[episode_begin : episode_end]).sum()
        blocks_per_episode.append(episode_blocks)
        episode_begin = episode_end

    return pd.DataFrame({"step": episode_ending_frames, "blocks": blocks_per_episode, "rewards": rewards_data["value"]})


def plot_folders(dpath_list):
    colors = "bgrcmyk"
    data_list = [get_rewards(path) for path in dpath_list]
    for data, path, c in zip(data_list, dpath_list, colors):
        #plt.plot(data["step"], data["blocks"], "x--", alpha=0.1)
        plt.plot(data["step"], smooth(data["blocks"], 0.9), c+"-",label=path+"_blocks")
        plt.plot(data["step"], smooth(data["rewards"], 0.9), c+"--",label=path+"_rewards")

    plt.legend()
    plt.show()

def display(dpath_list):

    for dpath in dpath_list:
        data = get_rewards(dpath)
        plt.plot(data["step"], data["blocks"], "rx--",alpha=0.1)
        plt.plot(data["step"], smooth(data["blocks"], 0.9), "r")
        plt.plot(data["step"], data["rewards"], "bx--", alpha=0.1)
        plt.plot(data["step"], smooth(data["rewards"], 0.9), "b")
    plt.ylim([0, 20])
    plt.plot(*average_line(dpath_list, "blocks"), "r-", linewidth=5, alpha=0.4, label="blocks")
    plt.plot(*average_line(dpath_list, "rewards"), "b-",linewidth=5, alpha=0.4, label="rewards")
    plt.legend()
    plt.show()


def average_line(dpath_list, tag="blocks"):
    # tag = "blocks" or "rewards"
    interp_blocks_list = []
    points = None
    for dpath in dpath_list:
        data = get_rewards(dpath)
        last_point = data["step"].iloc[-1]
        points = np.arange(0, last_point, 20)
        interp_blocks = np.interp(points, data["step"], smooth(data[tag], 0.9))
        interp_blocks_list.append(interp_blocks)

    smallest_len = min([len(bl) for bl in interp_blocks_list])
    interp_blocks_list = [interp_blocks[:smallest_len] for interp_blocks in interp_blocks_list]
    matrix_interp_blocks = np.array(interp_blocks_list)
    avr_interp_blocks = matrix_interp_blocks.sum(axis=0) / len(interp_blocks_list)
    points = points[:smallest_len]

    return points, avr_interp_blocks


def smooth(scalars, weight): # Weight between 0 and 1 
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


if __name__ == '__main__':
    #path = "path_to_your_summaries"
    #to_csv(path)
    path = "tensor/DQN_18"
    path_list = ["tensor/DQN_18", "tensor/DQN_28", "tensor/DQN_24", "tensor/DQN_22"]
    folders = ["tensor_t/DQN_1", "tensor_t/DQN_3", "tensor_t/DQN_5"]
    #folders = ["tensor_t/DQN_1", "tensor_t/DQN_3"]
    plot_folders(folders)
    #display(path_list)
    #plt.plot(*average_line(path_list, "blocks"), 'b')
    #plt.plot(*average_line(path_list, "rewards"), 'r')
    #plt.show()
    #path_list = ["tensor/DQN_18", "tensor/DQN_28", "tensor/DQN_24", "tensor/DQN_22"]
    #average_line(path_list)

