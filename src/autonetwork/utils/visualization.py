import numpy as np
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt

EpisodeStats = namedtuple(
    "Stats", ["episode_lengths", "episode_rewards", "episode_running_variance"]
)
TimestepStats = namedtuple("Stats", ["cumulative_rewards", "regrets"])


def plot_episode_stats(stats, smoothing_window=10, hideplot=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    if hideplot:
        plt.close(fig1)
    else:
        plt.show(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10, 5))
    rewards_smoothed = (
        pd.Series(stats.episode_rewards)
        .rolling(smoothing_window, min_periods=smoothing_window)
        .mean()
    )
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title(
        "Episode Reward over Time (Smoothed over window size {})".format(
            smoothing_window
        )
    )
    if hideplot:
        plt.close(fig2)
    else:
        plt.show(fig2)

    return fig1, fig2
