from functools import partial
import pretrained
from smacv2.env import MultiAgentEnv, StarCraft2Env
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper 
import sys
import os
import gym
from gym import ObservationWrapper, spaces
from gym.envs import registry as gym_registry
from gym.spaces import flatdim
import numpy as np
from gym.wrappers import TimeLimit as GymTimeLimit

# import lbforaging

import mpe

# Register custom envs.
gym_registry.register(
    id="Foraging-2s-20x20-2p-4f-coop-grid-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 2,
        "max_player_level": 3,
        "field_size": (20, 20),
        "max_food": 4,
        "sight": 2,
        "max_episode_steps": 50,
        "force_coop": True,
        "grid_observation": True,
    },
)
gym_registry.register(
    id="Foraging-2s-10x10-2p-4f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 2,
        "max_player_level": 3,
        "field_size": (10, 10),
        "max_food": 4,
        "sight": 2,
        "max_episode_steps": 50,
        "force_coop": True,
        "grid_observation": False,
    },
)
gym_registry.register(
    id="Foraging-2s-20x20-2p-1f-coop-grid-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 2,
        "max_player_level": 3,
        "field_size": (20, 20),
        "max_food": 1,
        "sight": 2,
        "max_episode_steps": 50,
        "force_coop": True,
        "grid_observation": True,
    },
)
gym_registry.register(
    id="Foraging-2s-20x20-2p-4f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 2,
        "max_player_level": 3,
        "field_size": (20, 20),
        "max_food": 4,
        "sight": 2,
        "max_episode_steps": 50,
        "force_coop": True,
        "grid_observation": False,
    },
)
gym_registry.register(
    id="Foraging-2s-20x20-2p-1f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 2,
        "max_player_level": 3,
        "field_size": (20, 20),
        "max_food": 1,
        "sight": 2,
        "max_episode_steps": 50,
        "force_coop": True,
        "grid_observation": False,
    },
)

gym_registry.register(
    id="Foraging-2s-8x8-2p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 2,
        "max_player_level": 2,
        "field_size": (8, 8),
        "max_food": 2,
        "sight": 2,
        "max_episode_steps": 30,
        "force_coop": True,
        "grid_observation": False,
    },
)
gym_registry.register(
    id="Foraging-2s-15x15-2p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 2,
        "max_player_level": 2,
        "field_size": (15, 15),
        "max_food": 2,
        "sight": 2,
        "max_episode_steps": 30,
        "force_coop": True,
        "grid_observation": False,
    },
)
gym_registry.register(
    id="Foraging-2s-15x15-2p-4f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 2,
        "max_player_level": 2,
        "field_size": (15, 15),
        "max_food": 4,
        "sight": 2,
        "max_episode_steps": 50,
        "force_coop": True,
        "grid_observation": False,
    },
)
gym_registry.register(
    id="Foraging-2s-15x15-4p-4f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 2,
        "field_size": (15, 15),
        "max_food": 4,
        "sight": 2,
        "max_episode_steps": 50,
        "force_coop": True,
        "grid_observation": False,
    },
)


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraftCapabilityEnvWrapper)

if sys.platform == "linux":
    os.environ.setdefault(
        "SC2PATH", os.path.join(os.getcwd(), "3rdparty", "StarCraftII")
    )


class TimeLimit(GymTimeLimit):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        # if self.env.spec is not None:
        #     self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert (
            self._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not all(done)
            done = len(observation) * [True]
        return observation, reward, done, info


class FlattenObservation(ObservationWrapper):
    r"""Observation wrapper that flattens the observation of individual agents."""

    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)

        ma_spaces = []

        for sa_obs in env.observation_space:
            flatdim = spaces.flatdim(sa_obs)
            ma_spaces += [
                spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(flatdim,),
                    dtype=np.float32,
                )
            ]

        self.observation_space = spaces.Tuple(tuple(ma_spaces))

    def observation(self, observation):
        return tuple(
            [
                spaces.flatten(obs_space, obs)
                for obs_space, obs in zip(self.env.observation_space, observation)
            ]
        )


class _GymmaWrapper(MultiAgentEnv):
    def __init__(self, key, time_limit, pretrained_wrapper, **kwargs):
        self.episode_limit = time_limit
        self._env = TimeLimit(gym.make(f"{key}"), max_episode_steps=time_limit)
        self._env = FlattenObservation(self._env)

        if pretrained_wrapper:
            self._env = getattr(pretrained, pretrained_wrapper)(self._env)

        self.n_agents = self._env.n_agents
        self._obs = None

        self.longest_action_space = max(self._env.action_space, key=lambda x: x.n)
        self.longest_observation_space = max(
            self._env.observation_space, key=lambda x: x.shape
        )

        self._seed = kwargs["seed"]
        self._env.seed(self._seed)

    def step(self, actions):
        """ Returns reward, terminated, info """
        actions = [int(a) for a in actions]
        self._obs, reward, done, info = self._env.step(actions)
        self._obs = [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in self._obs
        ]

        return float(sum(reward)), all(done), {}

    def get_obs(self):
        """ Returns all agent observations in a list """
        return self._obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise self._obs[agent_id]
    
    def get_capabilities(self):
        """Returns the capabilities of all agents in a list."""
        return None

    def get_capabilities_agent(self, agent_id):
        """Returns the capabilities of a single agent."""
        return None

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return flatdim(self.longest_observation_space)

    def get_state(self):
        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self):
        """ Returns the shape of the state"""
        return self.n_agents * flatdim(self.longest_observation_space)

    def get_cap_size(self):
        """Returns the size of the own capabilities of the agent."""
        return None

    def get_state(self):
        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self):
        """ Returns the shape of the state"""
        return self.n_agents * flatdim(self.longest_observation_space)

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        valid = flatdim(self._env.action_space[agent_id]) * [1]
        invalid = [0] * (self.longest_action_space.n - len(valid))
        return valid + invalid

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return flatdim(self.longest_action_space)

    def reset(self):
        """ Returns initial observations and states"""
        self._obs = self._env.reset()
        self._obs = [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in self._obs
        ]
        return self.get_obs(), self.get_state()

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    def seed(self):
        return self._env.seed

    def save_replay(self):
        pass

    def get_stats(self):
        return {}


REGISTRY["gymma"] = partial(env_fn, env=_GymmaWrapper)
