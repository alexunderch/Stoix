# Taken and modified from https://github.com/instadeepai/sebulba
from typing import Sequence

import chex
import jax
import numpy as np
from omegaconf import DictConfig

from stoix.base_types import Observation
from stoix.systems.sebulba import core
from stoix.systems.sebulba.metrics import MetricHub, RecordTimeTo


class AsyncActor(core.StoppableComponent):
    """Actor Component that generates trajectories by passing a batch of actions
    to our batched environment, which then outputs the corresponding observations. These
    are then used to form trajectories that are placed in our pipeline to be feed to our
    `Learner`.
    """

    def __init__(
        self,
        env_builder: core.EnvBuilder,
        actor_device: jax.Device,
        params_source: core.ParamsSource,
        pipeline: core.Pipeline,
        actor_fn: core.SebulbaActorFn,
        key: chex.PRNGKey,
        config: DictConfig,
        metrics_hub: MetricHub,
        name: str,
    ):
        """
        Create an `Actor` component that will initialise a batch of environments and pipeline to
        pass trajectories to a `Learner` component.

        Args:
            env_builder: An Callable that take an int and return a vector env
            actor_device: The jax device to use for the policy
            params_source: A source of parameter
            pipeline: The pipeline to put trajectories
            actor_fn: An ActorFn to generate action from observations
            key: A PRNGKey for the jax computations
            config: The Actor configuration
            metrics_hub: A hub to log metrics
            name: The name of this actor, used for thread name and logging path
        Returns:
            An actor that you can `start`, `stop` and `join`
        """
        super().__init__(name=f"Actor-{name}")
        self.actor_device = actor_device
        self.pipeline = pipeline
        self.params_source = params_source
        self.actor_fn = jax.jit(actor_fn)
        self.rng = jax.device_put(key, actor_device)
        self.config = config
        self.metrics_hub = metrics_hub
        self.cpu = jax.devices("cpu")[0]
        self.split_key_fn = jax.jit(jax.random.split)

        self.envs = env_builder(self.config.arch.actor.envs_per_actor)

    def _run(self) -> None:
        episode_return = np.zeros((self.config.arch.actor.envs_per_actor,))
        with jax.default_device(self.actor_device):

            obs, _ = self.envs.reset()
            obs = jax.device_put(obs, self.actor_device)

            while not self.should_stop:
                traj_obs = []
                traj_term = []
                traj_trunc = []
                traj_actions = []
                traj_extras = []
                traj_rewards = []

                for _t in range(self.config.system.rollout_length):
                    params = self.params_source.get()
                    with RecordTimeTo(self.metrics_hub["compute_action_time"]):
                        self.rng, key = self.split_key_fn(self.rng)
                        action, extra = self.actor_fn(params, obs, key)

                    with RecordTimeTo(self.metrics_hub["put_action_on_cpu_time"]):
                        action_cpu = np.array(jax.device_put(action, self.cpu))

                    with RecordTimeTo(self.metrics_hub["env_step_time"]):
                        next_obs, reward, terminated, truncated, info = self.envs.step(action_cpu)

                    episode_return += reward

                    traj_obs.append(obs)
                    traj_term.append(terminated)
                    traj_trunc.append(truncated)
                    traj_actions.append(action)
                    traj_extras.append(extra)
                    traj_rewards.append(reward)

                    dones = terminated | truncated
                    for env_idx, env_done in enumerate(dones):
                        if env_done:
                            self.metrics_hub["episode_return"].append(episode_return[env_idx])
                            self.metrics_hub["episode_length"].append(info["elapsed_step"][env_idx])
                    self.metrics_hub["number_of_episodes"].add(np.sum(dones))
                    # Reset the episode return when the episode is done
                    episode_return *= 1.0 - dones
                    # Set the new observation
                    obs = next_obs

                # Log the number of steps
                self.metrics_hub["steps"].add(self.config.arch.actor.envs_per_actor * self.config.system.rollout_length)
                # Send rollout to pipeline
                self.process_item(traj_obs, traj_term, traj_trunc, traj_actions, traj_extras, traj_rewards, obs)

    def process_item(
        self,
        traj_obs: Sequence[Observation],
        traj_term: Sequence[jax.Array],
        traj_trunc: Sequence[jax.Array],
        traj_actions: Sequence[jax.Array],
        traj_extras: Sequence[jax.Array],
        traj_rewards: Sequence[jax.Array],
        next_obs: Observation,
    ) -> None:
        """Process a trajectory and put it in the pipeline.
        All data here is a list of data in the shape (rollout_length, envs_per_actor, ...)
        except for next_obs which is in the shape (envs_per_actor, ...).
        """
        with RecordTimeTo(self.metrics_hub["pipeline_put_time"]):
            self.pipeline.put(traj_obs, traj_term, traj_trunc, traj_actions, traj_extras, traj_rewards, next_obs)
