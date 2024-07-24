import chex
import jax
import numpy as np
from omegaconf import DictConfig

from stoix.systems.sebulba import core
from stoix.systems.sebulba.metrics import MetricHub


class AsyncEvaluator(core.StoppableComponent):
    """Evaluator Component"""

    def __init__(
        self,
        env_builder: core.EnvBuilder,
        evaluator_device: jax.Device,
        params_source: core.ParamsSource,
        actor_fn: core.SebulbaActorFn,
        key: chex.PRNGKey,
        config: DictConfig,
        metrics_hub: MetricHub,
        name: str,
    ):
        """
        Create an `Evaluator` component that will initialise a batch of environments

        Args:
            env_builder: An Callable that take an int and return a vector env
            evaluator_device: The jax device to use for the policy
            params_source: A source of parameter
            actor_fn: An ActorFn to generate action from observations
            key: A PRNGKey for the jax computations
            config: The Actor configuration
            metrics_hub: A hub to log metrics
            name: The name of this actor, used for thread name and logging path
        Returns:
            An actor that you can `start`, `stop` and `join`
        """
        super().__init__(name=f"Evaluator-{name}")
        self.evaluator_device = evaluator_device
        self.params_source = params_source
        self.actor_fn = jax.jit(actor_fn)
        self.rng = jax.device_put(key, evaluator_device)
        self.config = config
        self.metrics_hub = metrics_hub
        self.cpu = jax.devices("cpu")[0]
        self.split_key_fn = jax.jit(jax.random.split)

        self.envs = env_builder(self.config.arch.num_eval_episodes)

    def _run(self) -> None:
        episode_return = np.zeros((self.config.arch.num_eval_episodes,))
        with jax.default_device(self.evaluator_device):

            obs, _ = self.envs.reset()
            obs = jax.device_put(obs, self.evaluator_device)

            while not self.should_stop:
                params = self.params_source.get()

                self.rng, key = self.split_key_fn(self.rng)
                action, _ = self.actor_fn(params, obs, key)

                action_cpu = np.array(jax.device_put(action, self.cpu))

                next_obs, reward, terminated, truncated, info = self.envs.step(action_cpu)

                episode_return += reward

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
