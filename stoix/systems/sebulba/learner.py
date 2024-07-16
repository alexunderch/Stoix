# Taken and modified from https://github.com/instadeepai/sebulba
import queue
from typing import Callable, List, Union

import chex
import jax

from stoix.base_types import StoixState
from stoix.systems.sebulba import core


class Learner(core.StoppableComponent):
    """
    `Learner` component, that retrieves trajectories from the `Pipeline` that are then used to
    carry out a learning update and updating the parameters of the `Actor`s.
    """

    def __init__(
        self,
        pipeline: core.Pipeline,
        local_devices: List[jax.Device],
        global_devices: List[jax.Device],
        init_state: StoixState,
        step_fn: core.LearnFn,
        key: chex.PRNGKey,
        on_params_change: Union[List[Callable], None] = None,
    ):
        """Creates a `Learner` component that will shard its state across the given devices. The
        given step_fn is wrapped in a `pmap` to allow for batched learning across the devices.

        Args:
            pipeline: A pipeline to get trajectories from
            local_devices: local devices to use for learner
            global_devices: global devices that are part of the learning
            init_state: the initial state of the algorithm
            step_fn: the function to pmap that define the learning
            key: A PRNGKey for the jax computations
            metrics_logger: a logger to log to
            on_params_change: a list of callable to call when there is new params
                (this is typically used to update Actors params)
        Returns:
            A Learner that you can `start`, `stop` and `join`.
        """
        super().__init__(name="Learner")
        self.pipeline = pipeline
        self.local_devices = local_devices
        self.global_devices = global_devices
        self.state = jax.device_put_replicated(init_state, self.local_devices)
        self.step_fn_pmaped = jax.pmap(
            step_fn,
            "batch",
            devices=global_devices,
            in_axes=(0, 0, None),  # type: ignore
        )
        self.on_params_change = on_params_change
        self.rng = key

    def _run(self) -> None:
        step = 0

        while not self.should_stop:
            try:
                batch = self.pipeline.get(block=True, timeout=1)
            except queue.Empty:
                continue
            else:
                # with logging.RecordTimeTo(self.metrics_logger["step_time"]):
                self.rng, key = jax.random.split(self.rng)
                self.state, metrics = self.step_fn_pmaped(self.state, batch, key)

                # jax.tree_util.tree_map_with_path(
                #     lambda path, value: self.metrics_logger[
                #         f"agents/{'/'.join([p.key for p in path])}"
                #     ].append(value[0].item()),
                #     metrics,
                # )

                if self.on_params_change is not None:
                    new_params = jax.tree_map(lambda x: x[0], self.state.params)
                    for handler in self.on_params_change:
                        handler(new_params)

                step += 1

                # self.metrics_logger["iteration"].add(1)
                # self.metrics_logger["steps"].add(math.prod(batch.actions.shape))
                # self.metrics_logger["queue_size"].append(self.pipeline.qsize())
