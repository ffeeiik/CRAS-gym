# -*- coding: utf-8 -*-


#每一条episode数据都是100个用户及他们的action
#观测环境中必须只能看到当前用户及之前的行为
"""Examples for recommender system simulating envs ready to be used by RLlib Algorithms.

This env follows RecSim obs and action APIs.
"""
import gymnasium as gym
import numpy as np
from typing import Optional

from ray.rllib.utils.numpy import softmax


class UserEnv(gym.Env):
   
    def __init__(self, num_users=100, num_actions=30, max_total_cost=1000, reward_dataset=None, cost_dataset=None):
        self.num_users = num_users
        self.num_actions = num_actions
        self.max_total_cost = max_total_cost
        self.reward_dataset = reward_dataset
        self.cost_dataset = cost_dataset

        self.action_space = gym.spaces.Discrete(num_actions)
        self.observation_space = gym.spaces.Dict({
            "user": gym.spaces.Box(low=0, high=1, shape=(num_users,), dtype=np.float32),
            "budget": gym.spaces.Box(low=0, high=max_total_cost, shape=(1,), dtype=np.float32),
            "reward": gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        })

        self.users = np.zeros(num_users)
        self.total_cost = 0
        self.total_reward = 0

    def reset(self):
        self.users = np.zeros(self.num_users)
        self.total_cost = 0
        self.total_reward = 0
        return {
            "user": self.users,
            "budget": self.max_otal_cost - self.total_cost,
            "reward": self.total_reward
        }

    #在环境中向前执行一步,此时的actions纬度是1*100，是整个episode中所有的action集合
    def step(self, actions):
        rewards = np.zeros(self.num_users)
        costs = np.zeros(self.num_users)

        for i, action in enumerate(actions):
            reward = self.reward_dataset[i, action]
            cost = self.cost_dataset[i, action]
            rewards[i] = reward
            costs[i] = cost

        self.users += rewards  #每个user做出选择后获得的reward
        self.total_cost += np.sum(costs)
        self.total_reward += np.sum(rewards)

        done = self.total_cost >= self.max_total_cost

        obs = {
            "user": self.users,
            "budget": self.max_total_cost - self.total_cost,
            "reward": self.total_reward
        }

        return obs, rewards, costs, done, {}


if __name__ == "__main__":
    """Test RecommSys env with random actions for baseline performance."""
    reward_dataset = np.random.rand(100, 30)  # Replace with your reward dataset
    cost_dataset = np.random.rand(100, 30)  # Replace with your cost dataset

    env = UserEnv(
        num_users=100,
        num_actions=30,
        max_total_cost=1000,
        reward_dataset=reward_dataset,
        cost_dataset=cost_dataset
    )
    # env = gym.make('UserEnv-v0')

    # obs = env.reset()
    num_episodes = 0
    episode_rewards = []
    episode_reward = 0.0

    while num_episodes < 100:
        actions = np.random.randint(0, 30, size=env.num_users)
        obs, rewards, costs, done, _ = env.step(actions)

        episode_reward += np.sum(rewards)
        if done:
            print(f"episode reward = {episode_reward}")
            print("User rewards:", rewards)
            print("User costs:", costs)
            obs = env.reset()
            num_episodes += 1
            episode_rewards.append(episode_reward)
            episode_reward = 0.0

    print(f"Avg reward={np.mean(episode_rewards)}")