import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
# import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
# from custom_types import TensorBatch
# import wandb
import json

from sklearn.preprocessing import OneHotEncoder

TensorBatch = List[torch.Tensor]


# @dataclass
class TrainConfig:
    # # wandb project name
    # project: str = "CORL"
    # # wandb group name
    # group: str = "BC-D4RL"
    # wandb run name
    # name: str = "BC"
    # training dataset and evaluation environment
    # env: str = "halfcheetah-medium-expert-v2"
    # total gradient updates during training
    max_timesteps: int = int(1e6)
    # training batch size
    batch_size: int = 256
    # maximum size of the replay buffer
    buffer_size: int = 9000
    # what top fraction of the dataset (sorted by return) to use
    frac: float = 0.1
    # maximum possible trajectory length
    max_traj_len: int = 1000
    # whether to normalize states
    normalize: bool = True
    # discount factor
    discount: float = 0.99
    # evaluation frequency, will evaluate eval_freq training steps
    eval_freq: int = int(5e3)
    # number of episodes to run during evaluation
    n_episodes: int = 10
    # path for checkpoints saving, optional
    checkpoints_path: Optional[str] = '/ossfs/workspace'
    # file name for loading a model, optional
    load_model: str = ""
    # training random seed
    # seed: int = 0
    # training device
    device: str = "cuda"

    # def __post_init__(self):
    #     self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
    #     if self.checkpoints_path is not None:
    #         self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


# def wrap_env(
#     env: gym.Env,
#     state_mean: Union[np.ndarray, float] = 0.0,
#     state_std: Union[np.ndarray, float] = 1.0,
#     reward_scale: float = 1.0,
# ) -> gym.Env:
#     # PEP 8: E731 do not assign a lambda expression, use a def
#     def normalize_state(state):
#         return (
#             state - state_mean
#         ) / state_std  # epsilon should be already added in std.

#     def scale_reward(reward):
#         # Please be careful, here reward is multiplied by scale!
#         return reward_scale * reward

#     env = gym.wrappers.TransformObservation(env, normalize_state)
#     if reward_scale != 1.0:
#         env = gym.wrappers.TransformReward(env, scale_reward)
#     return env


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device="cpu"
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device="cpu"
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device="cpu")
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device="cpu"
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device="cpu")
        # self._device = device="cpu"

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device="cpu")

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        print("n_transitions is:",n_transitions)
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError


# def set_seed(
#     seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
# ):
#     if env is not None:
#         env.seed(seed)
#         env.action_space.seed(seed)
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.manual_seed(seed)
#     torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


# @torch.no_grad()
# def eval_actor(
#     env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
# ) -> np.ndarray:
#     env.seed(seed)
#     actor.eval()
#     episode_rewards = []
#     for _ in range(n_episodes):
#         state, done = env.reset(), False
#         episode_reward = 0.0
#         while not done:
#             action = actor.act(state, device)
#             state, reward, done, _ = env.step(action)
#             episode_reward += reward
#         episode_rewards.append(episode_reward)

#     actor.train()
#     return np.asarray(episode_rewards)


def keep_best_trajectories(
    dataset: Dict[str, np.ndarray],
    frac: float,
    discount: float,
    max_episode_steps: int = 1000,
):
    ids_by_trajectories = []
    returns = []
    cur_ids = []
    cur_return = 0
    reward_scale = 1.0
    for i, (reward, done) in enumerate(zip(dataset["rewards"], dataset["terminals"])):
        cur_return += reward_scale * reward
        cur_ids.append(i)
        reward_scale *= discount
        if done == 1.0 or len(cur_ids) == max_episode_steps:
            ids_by_trajectories.append(list(cur_ids))
            returns.append(cur_return)
            cur_ids = []
            cur_return = 0
            reward_scale = 1.0

    sort_ord = np.argsort(returns, axis=0)[::-1].reshape(-1)
    top_trajs = sort_ord[: max(1, int(frac * len(sort_ord)))]

    order = []
    for i in top_trajs:
        order += ids_by_trajectories[i]
    order = order[0:8000]
    order = np.array(order)
    dataset["observations"] = dataset["observations"][order]
    print("actions is:", dataset["actions"].shape)
    dataset["actions"] = dataset["actions"][order]
    print("best actions is:", dataset["actions"].shape)

    dataset["next_observations"] = dataset["next_observations"][order]
    dataset["rewards"] = dataset["rewards"][order]
    dataset["terminals"] = dataset["terminals"][order]


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()


class BC:
    def __init__(
        self,
        max_action: np.ndarray,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        discount: float = 0.99,
        device: str = "cpu",
    ):
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.max_action = max_action
        self.discount = discount

        self.total_it = 0
        self.device = device

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        log_dict = {}
        self.total_it += 1

        state, action, _, _, _ = batch

        # Compute actor loss
        pi = self.actor(state)
        actor_loss = F.mse_loss(pi, action)
        log_dict["actor_loss"] = actor_loss.item()
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.total_it = state_dict["total_it"]


def state_processing(dataset):


    ad_position_data = dataset["observations"][-1]  # 广告展位序号
    ad_position_data = ad_position_data.reshape(-1, 1)
    next_ad_position_data = dataset["next_observations"][-1]
    next_ad_position_data = next_ad_position_data.reshape(-1, 1)
    encoder = OneHotEncoder(sparse=False)
    #(4e6, 4)
    one_hot_labels = encoder.fit_transform(ad_position_data)
    next_one_hot_labels = encoder.fit_transform(next_ad_position_data)

    observations_T = dataset["observations"][0:-1].T
    next_observations_T = dataset["next_observations"][0:-1].T

    # 将连续数据和广告展位序号的离散嵌入向量拼接在一起
    dataset["observations"] = np.concatenate([observations_T, one_hot_labels], axis = 1)
    dataset["next_observations"] = np.concatenate([next_observations_T, next_one_hot_labels], axis = 1)
    #处理连续状态值
    # for i in [0, 1, 2, 3, 4]:
    print("==========dataset_observations:", dataset["observations"].shape)
    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
        )
    print("-----observations:", dataset["observations"].shape)

    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
        )
    print("=======next_observations:", dataset["next_observations"].shape)
    # dataset["observations"] = torch.tensor(dataset["observations"], dtype=torch.float32) 
    # dataset["next_observations"] = torch.tensor(dataset["next_observations"], dtype=torch.float32)
    
    # user_data = user_data.unsqueeze(1)
    

    return dataset

def action_processing(dataset):

    encoder = OneHotEncoder(sparse=False)
    dataset["actions"] = encoder.fit_transform(dataset["actions"].reshape(-1, 1))

    return dataset


# @pyrallis.wrap()
def train(config: TrainConfig, dataset):

    
    dataset = state_processing(dataset)
    dataset = action_processing(dataset)
    keep_best_trajectories(dataset, config.frac, config.discount)


    state_dim = dataset["observations"].shape[1]
    print("-----------state dim is: ",state_dim)
    action_dim = 15

    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )

    replay_buffer.load_d4rl_dataset(dataset)

    max_action = 14

    # Set seeds
    # seed = config.seed
    # set_seed(seed, env)

    actor = Actor(state_dim, action_dim, max_action).to(config.device)
    #梯度累积
    # accumulation_steps = 4  # 尝试增大这个值
    # for t in range(int(config.max_timesteps)):
    #     for accumulation_step in range(accumulation_steps):
    #         batch = replay_buffer.sample(config.batch_size)
    #         batch = [b.to(config.device) for b in batch]
    #         trainer.train(batch)
    #     actor_optimizer.step()
    #     actor_optimizer.zero_grad()
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "discount": config.discount,
        "device": config.device,
    }

    print("---------------------------------------")
    # print(f"Training BC, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize policy
    trainer = BC(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    # wandb_init(asdict(config))

    # evaluations = []
    for t in range(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        trainer.train(batch)

    if config.checkpoints_path is not None:
        torch.save(
            trainer.state_dict(),
            os.path.join(config.checkpoints_path, f"checkpoint_final.pt"),
        )


if __name__ == "__main__":

# 读取JSON文件
    with open('/ossfs/workspace/output_incorrect_rank_ad_num.json', 'r') as file:
        data = json.load(file)
    # data = data[10000:20000]
    data["action"] = list(map(int, data["action"]))

    dataset = {
        "observations": np.array([data["user"]
        , data["current_budget"]
        ,data["flow_position"]
        
        ,data["time_slot"]
        ,data["rank_ad_num"],data["ad_position"]]
        ),

        "next_observations": np.array([data["next_user"]
        , data["budget"]
        ,data["next_flow_position"]
        
        ,data["next_time_slot"]
        ,data["next_rank_ad_num"],data["next_ad_position"]]),

        "actions": np.array(data["action"]),
        "rewards": np.array(data["reward"]),
        "terminals": np.array(data["terminals"])
    }
    
    config = TrainConfig(
    )
    train(config, dataset)
