import collections
import time
from abc import ABC, abstractmethod
from typing import Any, Deque, Dict, List, Union

import numpy as np

from stoix.systems.sebulba import core
from stoix.utils.logger import LogEvent, StoixLogger


class Metric(ABC):
    """Abstract class to define the interface of a metric.
    A metric is an object that stores data values and processes them for printing."""

    @abstractmethod
    def flush(self) -> Dict[str, float]:
        pass


class Scalar(Metric):
    """A simple class to store a scalar value and flush it.

    Practically, a queue is used to store the last 10 values and the time they were added.
    This allows us to compute the latest rate of the scalar and the average rate of the scalar
    over the last 10 values."""

    def __init__(self) -> None:
        self.value = 0
        self.queue: Deque = collections.deque(maxlen=10)
        self.flush_times: Deque = collections.deque(maxlen=10)

    def flush(self) -> Dict[str, float]:
        now = time.time()
        values = {
            "": float(self.value),
        }
        if len(self.queue) != 0:
            values["per_second"] = (self.value - self.queue[-1]) / (now - self.flush_times[-1])
            if len(self.queue) == 5:
                values["avg_per_second"] = (self.value - self.queue[0]) / (now - self.flush_times[0])

        self.queue.append(self.value)
        self.flush_times.append(now)
        return values

    def add(self, n: int) -> None:
        self.value += n


class StatisticsBetweenFlush(Metric):
    """A simple class to store a list of values and flush them. This differs from `Scalar` in that
    it stores all the values and computes the mean, min, max, and std of the values when flushed."""

    def __init__(self) -> None:
        self.values: List[float] = []

    def append(self, n: float) -> None:
        self.values.append(n)

    def flush(self) -> Dict[str, float]:
        values = self.values
        self.values = []
        statistics = {}
        if len(values) > 0:
            statistics["mean"] = float(np.mean(values))
            statistics["min"] = float(np.min(values))
            statistics["max"] = float(np.max(values))
            statistics["std"] = float(np.std(values))

        return statistics


class HubItem:
    """A class to store a metric and its parent. This class is used to store metrics in a hierarchical
    way, where a metric can have a parent metric. This allows us to store metrics in a tree-like structure."""

    def __init__(self, parent: Union[None, "HubItem"] = None) -> None:
        self.inner: Union[None, Scalar, StatisticsBetweenFlush] = None
        self.parent = parent

    def add(self, value: Any) -> None:
        """Add a value to a scalar metric. Additionally, if the metric has a parent,
        the value is added to the parent as well. If no scalar metric exists, a new
        scalar metric is created."""
        if self.inner is None:
            if self.parent is not None:
                self.parent.add(value)
            self.inner = Scalar()
        elif not isinstance(self.inner, Scalar):
            raise RuntimeError(f"This is not a scalar: {self.inner.__class__} your can't use add")
        elif self.parent is not None:
            self.parent.add(value)
        self.inner.add(value)

    def append(self, value: Any) -> None:
        """Add a value to a statistics metric. Additionally, if the metric has a parent,
        the value is added to the parent as well. If no statistics metric exists, a new
        statistics metric is created."""
        if self.inner is None:
            if self.parent is not None:
                self.parent.append(value)
            self.inner = StatisticsBetweenFlush()
        elif not isinstance(self.inner, StatisticsBetweenFlush):
            raise RuntimeError(f"This is not a StatisticsBetweenFlush: {self.inner.__class__} your can't use append")
        elif self.parent is not None:
            self.parent.append(value)
        self.inner.append(value)

    def create_sub(self) -> "HubItem":
        """Create a new sub metric. This is used to create a new metric in the hierarchy."""
        return HubItem(self)

    def flush(self) -> Dict[str, float]:
        if self.inner is None:
            return {}
        else:
            return self.inner.flush()


class MetricHub:
    """A class to store a set of metrics related to a parent process. Practically, each core process
    has its own metric hub. What this means is that the actors have their own metric hub and the
    learner has its own metric hub."""

    def __init__(self, name: str, parent: Union[None, "MetricHub"] = None) -> None:
        self.name = name
        self.metrics: Dict[str, HubItem] = {}
        self.parent = parent
        self.children: List["MetricHub"] = []

    def create_sub(self, name: str) -> "MetricHub":
        """Create a new metric hub and set it as a child of the current metric hub."""
        c = MetricHub(name, self)
        self.children.append(c)
        return c

    def __getitem__(self, key: str) -> HubItem:
        """Get a metric from the metric hub. If the metric does not exist, create a new metric."""
        if key in self.metrics:
            return self.metrics[key]
        if self.parent is not None:
            parrent_l = self.parent[key]
            metric = parrent_l.create_sub()
        else:
            metric = HubItem()
        self.metrics[key] = metric
        return metric

    def flush(self) -> Dict[str, float]:
        """Flush all the metrics in the metric hub."""
        values = {}
        names = list(self.metrics.keys())
        for name in names:
            metric = self.metrics[name]
            for name2, value in metric.flush().items():
                values[f"{self.name}_{name}_{name2}"] = value
        return values


class LoggerManager(core.StoppableComponent):
    """A class to manage the logging of metrics. This class stores metric hubs instead of
    metrics themselves. Additionally, it is the actual thread that controls the logging of
    metrics."""

    def __init__(self, stoix_logger: StoixLogger, logger_flush_dt: float = 1.5) -> None:
        super(LoggerManager, self).__init__()
        self.stoix_logger = stoix_logger
        self.logger_flush_dt = logger_flush_dt
        self.metric_hubs: Dict[str, MetricHub] = {}
        self.step = 0

    def __getitem__(self, name: str) -> MetricHub:
        """Get a metric hub. If the metric hub does not exist, create a new metric hub and return it."""
        if name in self.metric_hubs:
            return self.metric_hubs[name]
        c = MetricHub(name)
        self.metric_hubs[name] = c
        return c

    def flush(self) -> None:
        """Flush all the metrics in the metric hubs.
        We also specifically cater to metric hubs with the names `actors` and `learner`."""
        actor_values = {}
        learner_values = {}
        evaluator_values = {}
        misc_values = {}
        names = list(self.metric_hubs.keys())
        for name in names:
            metric_hub = self.metric_hubs[name]
            if "actors" in name:
                actor_values.update(metric_hub.flush())
            elif "learner" in name:
                learner_values.update(metric_hub.flush())
            elif "evaluator" in name:
                evaluator_values.update(metric_hub.flush())
            else:
                misc_values.update(metric_hub.flush())
        if actor_values:
            actor_values = {f"{k[7:]}": v for k, v in actor_values.items()}
            self.stoix_logger.log(actor_values, self.step, self.step, LogEvent.ACT)
        if learner_values:
            learner_values = {f"{k[8:]}": v for k, v in learner_values.items()}
            self.stoix_logger.log(learner_values, self.step, self.step, LogEvent.TRAIN)
        if evaluator_values:
            evaluator_values = {f"{k[10:]}": v for k, v in evaluator_values.items()}
            self.stoix_logger.log(evaluator_values, self.step, self.step, LogEvent.EVAL)
        if misc_values:
            self.stoix_logger.log(misc_values, self.step, self.step, LogEvent.MISC)
        if actor_values or learner_values or misc_values:
            self.step += 1

    def _run(self) -> None:
        """The main loop of the logger manager. This loop flushes the metrics every `logger_flush_dt` seconds."""
        before = time.time()
        while not self.should_stop:
            sleep_time = before - time.time() + self.logger_flush_dt
            if sleep_time > 0:
                time.sleep(sleep_time)
            before = time.time()
            self.flush()

        self.stoix_logger.stop()


class RecordTimeTo:
    """A context manager to record the time it takes to execute a block of code.
    This works specifically with hub items."""

    def __init__(self, to: HubItem):
        self.to = to

    def __enter__(self) -> None:
        self.start = time.monotonic()

    def __exit__(self, *args: Any) -> None:
        end = time.monotonic()
        self.to.append(end - self.start)
