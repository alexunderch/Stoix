# Taken and modified from https://github.com/instadeepai/sebulba
import math
import queue
from typing import Callable, List, Union

import chex
import jax

from stoix.base_types import StoixState
from stoix.systems.sebulba import core
from stoix.systems.sebulba.metrics import MetricHub, RecordTimeTo
from stoix.utils.jax_utils import unreplicate_device_dim


class AsyncLearner(core.StoppableComponent):
    """
    `Learner` component, that retrieves trajectories from the `Pipeline` that are then used to
    carry out a learning update and updating the parameters of the `Actor`s.
    """

    def __init__(
        self,
        pipeline: core.Pipeline,
        local_devices: List[jax.Device],
        init_state: StoixState,
        step_fn: core.LearnFn,
        key: chex.PRNGKey,
        metrics_hub: MetricHub,
        on_params_change: Union[List[Callable], None] = None,
    ):
        """Creates a `Learner` component that will shard its state across the given devices. The
        given step_fn is wrapped in a `pmap` to allow for batched learning across the devices.

        Args:
            pipeline: A pipeline to get trajectories from
            local_devices: local devices to use for learner
            init_state: the initial state of the algorithm
            step_fn: the function to pmap that define the learning
            key: A PRNGKey for the jax computations
            metrics_hub: a hub to log metrics
            on_params_change: a list of callable to call when there is new params
                (this is typically used to update Actors params)
        Returns:
            A Learner that you can `start`, `stop` and `join`.
        """
        super().__init__(name="Learner")
        self.pipeline = pipeline
        self.local_devices = local_devices
        self.state = jax.device_put_replicated(init_state, self.local_devices)
        self.step_fn_pmaped = jax.pmap(step_fn, axis_name="device", in_axes=(0, 0, None))
        self.on_params_change = on_params_change
        self.metrics_hub = metrics_hub
        self.rng = key

    def _run(self) -> None:
        step = 0

        while not self.should_stop:
            try:
                batch = self.pipeline.get(block=True, timeout=1)
            except queue.Empty:
                continue
            else:
                with RecordTimeTo(self.metrics_hub["step_time"]):
                    self.rng, key = jax.random.split(self.rng)
                    self.state, metrics = self.step_fn_pmaped(self.state, batch, key)

                    jax.tree_util.tree_map_with_path(
                        lambda path, value: self.metrics_hub[f"{'_'.join([p.key for p in path])}"].append(
                            value[0].item()
                        ),
                        metrics,
                    )

                if self.on_params_change is not None:
                    new_params = unreplicate_device_dim(self.state.params)
                    for handler in self.on_params_change:
                        handler(new_params)

                step += 1

                self.metrics_hub["iteration"].add(1)
                self.metrics_hub["steps"].add(math.prod(batch.actions.shape))
                self.metrics_hub["queue_size"].append(self.pipeline.qsize())
