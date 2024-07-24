import copy
import threading
from typing import Any, Dict, NamedTuple, Tuple

import chex
import envpool
import hydra
import jax
import jax.numpy as jnp
import omegaconf
import optax
from colorama import Fore, Style
from flax.core.frozen_dict import FrozenDict
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

from stoix.base_types import (
    ActorApply,
    ActorCriticOptStates,
    ActorCriticParams,
    CriticApply,
    ExperimentOutput,
    Extras,
    Observation,
    Parameters,
)
from stoix.evaluator import evaluator_setup, get_distribution_act_fn
from stoix.networks.base import FeedForwardActor as Actor
from stoix.networks.base import FeedForwardCritic as Critic
from stoix.systems.anakin.ppo.ppo_types import PPOTransition
from stoix.systems.sebulba import core
from stoix.systems.sebulba.actor import AsyncActor
from stoix.systems.sebulba.evaluator import AsyncEvaluator
from stoix.systems.sebulba.learner import AsyncLearner
from stoix.systems.sebulba.metrics import LoggerManager
from stoix.systems.sebulba.stoppers import ActorStepStopper, LearnerStepStopper, Stopper
from stoix.utils import make_env as environments
from stoix.utils.checkpointing import Checkpointer
from stoix.utils.env_pool import EnvPoolFactory
from stoix.utils.jax_utils import (
    merge_leading_dims,
    unreplicate_batch_dim,
    unreplicate_n_dims,
)
from stoix.utils.logger import LogEvent, StoixLogger
from stoix.utils.loss import clipped_value_loss, ppo_clip_loss
from stoix.utils.multistep import batch_truncated_generalized_advantage_estimation
from stoix.utils.total_timestep_checker import check_total_timesteps
from stoix.utils.training import make_learning_rate
from stoix.wrappers.episode_metrics import get_final_step_metrics

def get_learner_fn(
    apply_fns: Tuple[ActorApply, CriticApply],
    update_fns: Tuple[optax.TransformUpdateFn, optax.TransformUpdateFn],
    config: DictConfig,
) -> core.SebulbaLearnFn[core.CoreLearnerState]:
    """Get the learner function."""

    # Get apply and update functions for actor and critic networks.
    actor_apply_fn, critic_apply_fn = apply_fns
    actor_update_fn, critic_update_fn = update_fns

    def _update_step(
        learner_state: core.CoreLearnerState, traj_batch: PPOTransition, key: chex.PRNGKey
    ) -> Tuple[core.CoreLearnerState, Tuple]:

        # CALCULATE ADVANTAGE
        params, opt_states = learner_state

        r_t = traj_batch.reward
        v_t = traj_batch.value
        d_t = 1.0 - traj_batch.done.astype(jnp.float32)
        d_t = (d_t * config.system.gamma).astype(jnp.float32)
        advantages, targets = batch_truncated_generalized_advantage_estimation(
            r_t,
            d_t,
            config.system.gae_lambda,
            v_t,
            time_major=True,
            standardize_advantages=config.system.standardize_advantages,
            truncation_flags=traj_batch.truncated,
        )

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            """Update the network for a single epoch."""

            def _update_minibatch(train_state: Tuple, batch_info: Tuple) -> Tuple:
                """Update the network for a single minibatch."""

                # UNPACK TRAIN STATE AND BATCH INFO
                params, opt_states = train_state
                traj_batch, advantages, targets = batch_info

                def _actor_loss_fn(
                    actor_params: FrozenDict,
                    traj_batch: PPOTransition,
                    gae: chex.Array,
                ) -> Tuple:
                    """Calculate the actor loss."""
                    # RERUN NETWORK
                    actor_policy = actor_apply_fn(actor_params, traj_batch.obs)
                    log_prob = actor_policy.log_prob(traj_batch.action)

                    # CALCULATE ACTOR LOSS
                    loss_actor = ppo_clip_loss(log_prob, traj_batch.log_prob, gae, config.system.clip_eps)
                    entropy = actor_policy.entropy().mean()

                    total_loss_actor = loss_actor - config.system.ent_coef * entropy
                    loss_info = {
                        "actor_loss": loss_actor,
                        "entropy": entropy,
                    }
                    return total_loss_actor, loss_info

                def _critic_loss_fn(
                    critic_params: FrozenDict,
                    traj_batch: PPOTransition,
                    targets: chex.Array,
                ) -> Tuple:
                    """Calculate the critic loss."""
                    # RERUN NETWORK
                    value = critic_apply_fn(critic_params, traj_batch.obs)

                    # CALCULATE VALUE LOSS
                    value_loss = clipped_value_loss(value, traj_batch.value, targets, config.system.clip_eps)

                    critic_total_loss = config.system.vf_coef * value_loss
                    loss_info = {
                        "value_loss": value_loss,
                    }
                    return critic_total_loss, loss_info

                # CALCULATE ACTOR LOSS
                actor_grad_fn = jax.grad(_actor_loss_fn, has_aux=True)
                actor_grads, actor_loss_info = actor_grad_fn(params.actor_params, traj_batch, advantages)

                # CALCULATE CRITIC LOSS
                critic_grad_fn = jax.grad(_critic_loss_fn, has_aux=True)
                critic_grads, critic_loss_info = critic_grad_fn(params.critic_params, traj_batch, targets)

                # pmean over devices.
                actor_grads, actor_loss_info = jax.lax.pmean((actor_grads, actor_loss_info), axis_name="device")

                # pmean over devices.
                critic_grads, critic_loss_info = jax.lax.pmean((critic_grads, critic_loss_info), axis_name="device")

                # UPDATE ACTOR PARAMS AND OPTIMISER STATE
                actor_updates, actor_new_opt_state = actor_update_fn(actor_grads, opt_states.actor_opt_state)
                actor_new_params = optax.apply_updates(params.actor_params, actor_updates)

                # UPDATE CRITIC PARAMS AND OPTIMISER STATE
                critic_updates, critic_new_opt_state = critic_update_fn(critic_grads, opt_states.critic_opt_state)
                critic_new_params = optax.apply_updates(params.critic_params, critic_updates)

                # PACK NEW PARAMS AND OPTIMISER STATE
                new_params = ActorCriticParams(actor_new_params, critic_new_params)
                new_opt_state = ActorCriticOptStates(actor_new_opt_state, critic_new_opt_state)

                # PACK LOSS INFO
                loss_info = {
                    **actor_loss_info,
                    **critic_loss_info,
                }
                return (new_params, new_opt_state), loss_info

            params, opt_states, traj_batch, advantages, targets, key = update_state
            key, shuffle_key = jax.random.split(key)

            # SHUFFLE MINIBATCHES
            # Since we shard the envs per actor across the devices
            envs_per_batch = config.arch.actor.envs_per_actor // len(config.arch.learner.device_ids)
            batch_size = config.system.rollout_length * envs_per_batch
            permutation = jax.random.permutation(shuffle_key, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = jax.tree_util.tree_map(lambda x: merge_leading_dims(x, 2), batch)
            shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, [config.system.num_minibatches, -1] + list(x.shape[1:])),
                shuffled_batch,
            )

            # UPDATE MINIBATCHES
            (params, opt_states), loss_info = jax.lax.scan(_update_minibatch, (params, opt_states), minibatches)

            update_state = (params, opt_states, traj_batch, advantages, targets, key)

            return update_state, loss_info

        update_state = (params, opt_states, traj_batch, advantages, targets, key)

        # UPDATE EPOCHS
        update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config.system.epochs)

        params, opt_states, traj_batch, advantages, targets, key = update_state
        learner_state = core.CoreLearnerState(params, opt_states)

        return learner_state, loss_info

    def learner_fn(
        learner_state: core.CoreLearnerState, traj_batch: core.BaseTrajectory, key
    ) -> ExperimentOutput[core.CoreLearnerState]:

        values = traj_batch.extras["values"]
        bootstrap_value = critic_apply_fn(learner_state.params.critic_params, traj_batch.next_obs)
        values = jnp.concatenate([values, bootstrap_value[jnp.newaxis, ...]], axis=0)

        traj_batch = PPOTransition(
            done=traj_batch.done,
            obs=traj_batch.obs,
            action=traj_batch.action,
            reward=traj_batch.reward,
            value=values,
            log_prob=traj_batch.extras["log_probs"],
            truncated=traj_batch.done,
            info=traj_batch.extras,
        )
        learner_state, loss_info = _update_step(learner_state, traj_batch, key)

        def mean_all(x: jax.Array) -> jax.Array:
            return jax.lax.pmean(jnp.mean(x), axis_name="device")

        loss_info = jax.tree_util.tree_map(lambda x: mean_all(x), loss_info)

        return learner_state, loss_info

    return learner_fn


def learner_setup(
    env_factory, keys: Tuple[chex.PRNGKey, chex.PRNGKey, chex.PRNGKey], config: DictConfig
) -> Tuple[core.SebulbaLearnFn[core.CoreLearnerState], Actor, core.CoreLearnerState]:
    """Initialise learner_fn, network, optimiser"""

    # Get number/dimension of actions.
    env = env_factory(num_envs=1)
    obs_shape = env.observation_space.shape
    num_actions = int(env.action_space.n)
    env.close()
    config.system.action_dim = num_actions

    # PRNG keys.
    key, actor_net_key, critic_net_key = keys

    # Define network and optimiser.
    actor_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    actor_action_head = hydra.utils.instantiate(config.network.actor_network.action_head, action_dim=num_actions)
    critic_torso = hydra.utils.instantiate(config.network.critic_network.pre_torso)
    critic_head = hydra.utils.instantiate(config.network.critic_network.critic_head)

    actor_network = Actor(torso=actor_torso, action_head=actor_action_head)
    critic_network = Critic(torso=critic_torso, critic_head=critic_head)

    actor_lr = make_learning_rate(config.system.actor_lr, config, config.system.epochs, config.system.num_minibatches)
    critic_lr = make_learning_rate(config.system.critic_lr, config, config.system.epochs, config.system.num_minibatches)

    actor_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(actor_lr, eps=1e-5),
    )
    critic_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(critic_lr, eps=1e-5),
    )

    # Initialise observation
    init_x = jnp.ones(obs_shape)
    init_x = jax.tree_util.tree_map(lambda x: x[None, ...], init_x)

    # Initialise actor params and optimiser state.
    actor_params = actor_network.init(actor_net_key, init_x)
    actor_opt_state = actor_optim.init(actor_params)

    # Initialise critic params and optimiser state.
    critic_params = critic_network.init(critic_net_key, init_x)
    critic_opt_state = critic_optim.init(critic_params)

    # Pack params.
    params = ActorCriticParams(actor_params, critic_params)

    actor_network_apply_fn = actor_network.apply
    critic_network_apply_fn = critic_network.apply

    # Pack apply and update functions.
    apply_fns = (actor_network_apply_fn, critic_network_apply_fn)
    update_fns = (actor_optim.update, critic_optim.update)

    learn = get_learner_fn(apply_fns, update_fns, config)

    # Load model from checkpoint if specified.
    if config.logger.checkpointing.load_model:
        loaded_checkpoint = Checkpointer(
            model_name=config.system.system_name,
            **config.logger.checkpointing.load_args,  # Other checkpoint args
        )
        # Restore the learner state from the checkpoint
        restored_params, _ = loaded_checkpoint.restore_params()
        # Update the params
        params = restored_params

    # Define params to be replicated across devices.
    opt_states = ActorCriticOptStates(actor_opt_state, critic_opt_state)
    # Initialise learner state.
    init_learner_state = core.CoreLearnerState(params, opt_states)

    return learn, apply_fns, init_learner_state


def get_actor_fn(actor_apply_fn: ActorApply, critic_apply_fn: CriticApply) -> core.SebulbaActorFn:
    """Create the actor fn executed by the actor threads"""

    def actor_fn(params: ActorCriticParams, obs: Observation, key: chex.PRNGKey) -> Tuple[chex.Array, Extras]:
        pi = actor_apply_fn(params.actor_params, obs)
        action = pi.sample(seed=key)
        logprob = pi.log_prob(action)
        value = critic_apply_fn(params.critic_params, obs)
        extras = {
            "log_probs": logprob,
            "values": value.squeeze(),
        }
        return action, extras

    return actor_fn


def run_experiment(_config: DictConfig) -> float:
    """Runs experiment."""
    config = copy.deepcopy(_config)

    # Create environments factory
    # This creates a callable function that returns vectorised environments
    env_factory = environments.make_envpool_factory(config)
    
    # Get the learner and actor devices
    local_devices = jax.local_devices()
    global_devices = jax.devices()
    assert len(local_devices) == len(
        global_devices
    ), "Local and global devices must be the same for now. We dont support multihost just yet"

    actor_devices = [local_devices[device_id] for device_id in config.arch.actor.device_ids]
    local_learner_devices = [local_devices[device_id] for device_id in config.arch.learner.device_ids]
    evaluator_device = local_devices[config.arch.evaluator_device_id]
    config.num_learning_devices = len(local_learner_devices)
    config.num_actor_actor_devices = len(actor_devices)

    print(f"{Fore.YELLOW}{Style.BRIGHT}[Sebulba] Actors devices: {actor_devices}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{Style.BRIGHT}[Sebulba] Learner devices: {local_learner_devices}{Style.RESET_ALL}")

    # PRNG keys.
    key, actor_net_key, critic_net_key = jax.random.split(jax.random.PRNGKey(config.arch.seed), num=3)

    # Setup learner.
    learn, apply_fns, learner_state = learner_setup(env_factory, (key, actor_net_key, critic_net_key), config)

    # Setup the actor function
    actor_apply_fn, critic_apply_fn = apply_fns
    actor_fn = get_actor_fn(actor_apply_fn, critic_apply_fn)

    # Logger setup
    logger = StoixLogger(config)
    cfg: Dict = OmegaConf.to_container(config, resolve=True)
    cfg["arch"]["devices"] = jax.devices()
    pprint(cfg)

    # Set up the global logger manager
    # This creates the thread that manages the shared logging of metrics between threads
    global_logger_manager = LoggerManager(logger)
    # Start the logger manager thread
    global_logger_manager.start()

    # Build the Pipeline to pass trajectories between Actor's and Learner
    pipeline = core.Pipeline(max_size=5, learner_devices=local_learner_devices)
    pipeline.start()

    # Split the key for the actors and learner
    key, learner_key, actors_key = jax.random.split(key, 3)

    # Calculate the number of envs per actor
    num_envs_per_actor_device = config.arch.total_num_envs // len(actor_devices)
    num_envs_per_actor = num_envs_per_actor_device // config.arch.actor.actor_per_device
    config.arch.actor.envs_per_actor = num_envs_per_actor
    
    
    # Creating the actors and evaluator
    actors = []
    params_sources = []
    # Here we create a metric hub for the actors and evaluator using the global logger manager
    actors_loggers = global_logger_manager["actors"]
    eval_logger = global_logger_manager["evaluator"]
    
    # Create 1 params source for the evaluator
    params_source = core.ParamsSource(learner_state.params, evaluator_device)
    params_source.start()
    params_sources.append(params_source)
    eval_key, key = jax.random.split(key)
    # Create the evaluator
    evaluator = AsyncEvaluator(
        env_factory,
        evaluator_device,
        params_source,
        actor_fn,
        eval_key,
        config,
        eval_logger,
        f"Eval-{evaluator_device.id}",
        )
        
    for actor_device in actor_devices:
        # Create 1 params source per actor device as this will be used to pass the params to the actors
        params_source = core.ParamsSource(learner_state.params, actor_device)
        params_source.start()
        params_sources.append(params_source)
        # Now for each device we choose to create multiple actor threads
        for i in range(config.arch.actor.actor_per_device):
            actors_key, key = jax.random.split(actors_key)
            # Create the actual actor
            actor = AsyncActor(
                env_factory,
                actor_device,
                params_source,
                pipeline,
                actor_fn,
                key,
                config,
                actors_loggers,
                f"{actor_device.id}-{i}",
            )
            actors.append(actor)

    # Create a metric hub for the learner using the global logger manager
    learner_loggers = global_logger_manager["learner"]
    # Create Learner
    # Here we pass in all of the params_source update fns so that the
    # learner can update all the actors
    learner = AsyncLearner(
        pipeline,
        local_learner_devices,
        learner_state,
        learn,
        learner_key,
        learner_loggers,
        on_params_change=[params_source.update for params_source in params_sources],
    )


    # Now we start all the Learner and Actor threads
    learner.start()
    for actor in actors:
        actor.start()
    evaluator.start()
    # These are now running and performing their tasks separately
    # So in the main thread we can now wait for the stopper thread
    # to signal to them to stop running

    try:
        # Create our stopper and wait for it to stop
        # We pass it the global logger manager, not just a metric hub so it
        # has access to all metrics and can use them to decide when to stop
        stopper = ActorStepStopper(config, logger_manager=global_logger_manager)
        stopper.wait()
    finally:
        print(f"{Fore.RED}{Style.BRIGHT}Shutting down{Style.RESET_ALL}")

        # Try to gracefully shutdown all components
        # If not finished after 10 second exits anyway

        def graceful_shutdown():
            for actor in actors:
                actor.stop()
            evaluator.stop()
            for params_source in params_sources:
                params_source.stop()

            for actor in actors:
                actor.join()
            evaluator.join()

            learner.stop()
            learner.join()

            pipeline.stop()
            pipeline.join()

            global_logger_manager.stop()
            for params_source in params_sources:
                params_source.join()
            global_logger_manager.join()

        graceful_thread = threading.Thread(target=graceful_shutdown, daemon=True)
        graceful_thread.start()
        graceful_thread.join(timeout=10.0)
        if graceful_thread.is_alive():
            print(f"{Fore.RED}{Style.BRIGHT}Shutdown was not graceful{Style.RESET_ALL}")

    
    # Set up checkpointer
    # save_checkpoint = config.logger.checkpointing.save_model
    # if save_checkpoint:
    #     checkpointer = Checkpointer(
    #         metadata=config,  # Save all config as metadata in the checkpoint
    #         model_name=config.system.system_name,
    #         **config.logger.checkpointing.save_args,  # Checkpoint args
    #     )

    # Run experiment for a total number of evaluations.
    # max_episode_return = jnp.float32(-1e7)
    # best_params = unreplicate_batch_dim(learner_state.params.actor_params)
    # for eval_step in range(config.arch.num_evaluation):
    #     # Train.
    #     start_time = time.time()

    #     learner_output = learn(learner_state)
    #     jax.block_until_ready(learner_output)

    #     # Log the results of the training.
    #     elapsed_time = time.time() - start_time
    #     t = int(steps_per_rollout * (eval_step + 1))
    #     episode_metrics, ep_completed = get_final_step_metrics(learner_output.episode_metrics)
    #     episode_metrics["steps_per_second"] = steps_per_rollout / elapsed_time

    #     # Separately log timesteps, actoring metrics and training metrics.
    #     logger.log({"timestep": t}, t, eval_step, LogEvent.MISC)
    #     if ep_completed:  # only log episode metrics if an episode was completed in the rollout.
    #         logger.log(episode_metrics, t, eval_step, LogEvent.ACT)
    #     logger.log(learner_output.train_metrics, t, eval_step, LogEvent.TRAIN)

    #     # Prepare for evaluation.
    #     start_time = time.time()
    #     trained_params = unreplicate_batch_dim(
    #         learner_output.learner_state.params.actor_params
    #     )  # Select only actor params
    #     key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
    #     eval_keys = jnp.stack(eval_keys)
    #     eval_keys = eval_keys.reshape(n_devices, -1)

    #     # Evaluate.
    #     evaluator_output = evaluator(trained_params, eval_keys)
    #     jax.block_until_ready(evaluator_output)

    #     # Log the results of the evaluation.
    #     elapsed_time = time.time() - start_time
    #     episode_return = jnp.mean(evaluator_output.episode_metrics["episode_return"])

    #     steps_per_eval = int(jnp.sum(evaluator_output.episode_metrics["episode_length"]))
    #     evaluator_output.episode_metrics["steps_per_second"] = steps_per_eval / elapsed_time
    #     logger.log(evaluator_output.episode_metrics, t, eval_step, LogEvent.EVAL)

    #     if save_checkpoint:
    #         # Save checkpoint of learner state
    #         checkpointer.save(
    #             timestep=int(steps_per_rollout * (eval_step + 1)),
    #             unreplicated_learner_state=unreplicate_n_dims(learner_output.learner_state),
    #             episode_return=episode_return,
    #         )

    #     if config.arch.absolute_metric and max_episode_return <= episode_return:
    #         best_params = copy.deepcopy(trained_params)
    #         max_episode_return = episode_return

    #     # Update runner state to continue training.
    #     learner_state = learner_output.learner_state

    # # Measure absolute metric.
    # if config.arch.absolute_metric:
    #     start_time = time.time()

    #     key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
    #     eval_keys = jnp.stack(eval_keys)
    #     eval_keys = eval_keys.reshape(n_devices, -1)

    #     evaluator_output = absolute_metric_evaluator(best_params, eval_keys)
    #     jax.block_until_ready(evaluator_output)

    #     elapsed_time = time.time() - start_time
    #     t = int(steps_per_rollout * (eval_step + 1))
    #     steps_per_eval = int(jnp.sum(evaluator_output.episode_metrics["episode_length"]))
    #     evaluator_output.episode_metrics["steps_per_second"] = steps_per_eval / elapsed_time
    #     logger.log(evaluator_output.episode_metrics, t, eval_step, LogEvent.ABSOLUTE)

    # # Stop the logger.
    # logger.stop()
    # # Record the performance for the final evaluation run. If the absolute metric is not
    # # calculated, this will be the final evaluation run.
    # eval_performance = float(jnp.mean(evaluator_output.episode_metrics[config.env.eval_metric]))
    # return eval_performance


@hydra.main(config_path="../../../configs", config_name="default_ff_ppo.yaml", version_base="1.2")
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    eval_performance = run_experiment(cfg)

    print(f"{Fore.CYAN}{Style.BRIGHT}PPO experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
