import copy
import threading
import time
from typing import Any, Dict, NamedTuple, Tuple

import chex
import envpool
import flax
import hydra
import jax
import jax.numpy as jnp
import omegaconf
import optax
from colorama import Fore, Style
from flax.core.frozen_dict import FrozenDict
from jumanji.env import Environment
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

from stoix.base_types import (
    ActorApply,
    ActorCriticOptStates,
    ActorCriticParams,
    CriticApply,
    ExperimentOutput,
    LearnerFn,
    OptStates,
    Parameters,
)
from stoix.evaluator import evaluator_setup, get_distribution_act_fn
from stoix.networks.base import FeedForwardActor as Actor
from stoix.networks.base import FeedForwardCritic as Critic
from stoix.systems.anakin.ppo.ppo_types import PPOTransition
from stoix.systems.sebulba import core
from stoix.systems.sebulba.actor import AsyncActor
from stoix.systems.sebulba.learner import AsyncLearner
from stoix.systems.sebulba.logging import LoggerManager
from stoix.systems.sebulba.stoppers import LearnerStepStopper, Stopper
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

def get_env_specs(name: str) -> omegaconf.DictConfig:
    env = envpool.make(name, "gym")
    obs_shape = env.observation_space.shape
    n_action = env.action_space.n
    env.close()
    return omegaconf.OmegaConf.create(
        {
            "obs_shape": obs_shape,
            "n_action": n_action,
        }
    )

class CoreLearnerState(NamedTuple):
    params: Parameters
    opt_states: OptStates
    key: chex.PRNGKey

def get_learner_fn(
    apply_fns: Tuple[ActorApply, CriticApply],
    update_fns: Tuple[optax.TransformUpdateFn, optax.TransformUpdateFn],
    config: DictConfig,
) -> LearnerFn[CoreLearnerState]:
    """Get the learner function."""

    # Get apply and update functions for actor and critic networks.
    actor_apply_fn, critic_apply_fn = apply_fns
    actor_update_fn, critic_update_fn = update_fns

    def _update_step(learner_state: CoreLearnerState, traj_batch : PPOTransition) -> Tuple[CoreLearnerState, Tuple]:

        # CALCULATE ADVANTAGE
        params, opt_states, key= learner_state
        
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

                # Compute the parallel mean (pmean) over the batch.
                # This calculation is inspired by the Anakin architecture demo notebook.
                # available at https://tinyurl.com/26tdzs5x
                # This pmean could be a regular mean as the batch axis is on the same device.
                actor_grads, actor_loss_info = jax.lax.pmean((actor_grads, actor_loss_info), axis_name="batch")
                # pmean over devices.
                actor_grads, actor_loss_info = jax.lax.pmean((actor_grads, actor_loss_info), axis_name="device")

                critic_grads, critic_loss_info = jax.lax.pmean((critic_grads, critic_loss_info), axis_name="batch")
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
            batch_size = config.system.rollout_length * config.arch.num_envs
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
        learner_state = CoreLearnerState(params, opt_states, key)
        metric = traj_batch.info
        return learner_state, (metric, loss_info)

    def learner_fn(learner_state: CoreLearnerState, traj_batch : PPOTransition) -> ExperimentOutput[CoreLearnerState]:
        
        learner_state, (episode_info, loss_info) = _update_step(learner_state, traj_batch)
        
        return ExperimentOutput(
            learner_state=learner_state,
            episode_metrics=episode_info,
            train_metrics=loss_info,
        )

    return learner_fn


def learner_setup(
    keys: chex.Array, config: DictConfig
) -> Tuple[LearnerFn[CoreLearnerState], Actor, CoreLearnerState]:
    """Initialise learner_fn, network, optimiser, environment and states."""
    # Get available TPU cores.
    n_devices = len(jax.devices())

    # Get number/dimension of actions.
    num_actions = config.n_action
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
    init_x = jnp.ones(config.obs_shape)
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

    # Get batched iterated update and replicate it to pmap it over cores.
    learn = get_learner_fn(apply_fns, update_fns, config)
    learn = jax.pmap(learn, axis_name="device")

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

    # Define params to be replicated across devices and batches.
    key, step_key = jax.random.split(key)
    step_keys = jax.random.split(step_key, n_devices)
    step_keys = jnp.stack(step_keys)
    opt_states = ActorCriticOptStates(actor_opt_state, critic_opt_state)
    replicate_learner = (params, opt_states)

    # Duplicate learner across devices.
    replicate_learner = flax.jax_utils.replicate(replicate_learner, devices=jax.devices())

    # Initialise learner state.
    params, opt_states = replicate_learner
    init_learner_state = CoreLearnerState(params, opt_states, step_keys)

    return learn, actor_network, init_learner_state


def run_experiment(_config: DictConfig) -> float:
    """Runs experiment."""
    config = copy.deepcopy(_config)
    
    # Get jax devices local and global for learning
    local_devices = jax.local_devices()
    global_devices = jax.devices()
    assert len(local_devices) == len(global_devices), "Local and global devices must be the same for now."
    actor_devices = [local_devices[device_id] for device_id in config.arch.actor.device_ids]
    local_learner_devices = [
        local_devices[device_id] for device_id in config.arch.learner.device_ids
    ]
    print(f"{Fore.YELLOW}{Style.BRIGHT}[Sebulba] Actors devices: {actor_devices}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{Style.BRIGHT}[Sebulba] Learner devices: {local_learner_devices}{Style.RESET_ALL}")

    # Calculate total timesteps.
    n_devices = len(jax.devices())
    config.num_devices = n_devices
    config = check_total_timesteps(config)
    assert (
        config.arch.num_updates >= config.arch.num_evaluation
    ), "Number of updates per evaluation must be less than total number of updates."

    # PRNG keys.
    key, key_e, actor_net_key, critic_net_key = jax.random.split(jax.random.PRNGKey(config.arch.seed), num=4)

    # Setup learner.
    learn, actor_network, learner_state = learner_setup((key, actor_net_key, critic_net_key), config)
    
    # Logger setup
    logger = StoixLogger(config)
    cfg: Dict = OmegaConf.to_container(config, resolve=True)
    cfg["arch"]["devices"] = jax.devices()
    pprint(cfg)
    
    logger_manager = LoggerManager(logger)
    logger_manager.start()
    
    def actor_fn(params: Parameters, obs: chex.Array, key: chex.PRNGKey) -> Tuple[chex.Array, Any]:
        """Actor function."""
        return actor_network.apply(params.actor_params, obs)
    
    # Create the environments for train and eval.
    # Create environments factory, uses the same setup as seen in CleanRL
    env_factory = EnvPoolFactory(
        config.arch.seed,
        task_id=config.env.scenario.name,
        env_type="dm",
        **config.env.kwargs,
    )
    
    env_specs = get_env_specs(config.env.scenario.name)
    config = OmegaConf.merge(config, env_specs)
    
    # build Pipeline to pass trajectories between Actor's and Learner
    partial_pipeline = hydra.utils.instantiate(cfg.pipeline)
    pipeline: core.Pipeline = partial_pipeline(learner_devices=local_learner_devices)
    pipeline.start()
    
    key, learner_key, actors_key = jax.random.split(key, 3)
    
    actors = []
    params_sources = []
    for actor_device in actor_devices:
        # Create 1 params source per actor device
        params_source = core.ParamsSource(learner_state.params, actor_device)
        params_source.start()
        params_sources.append(params_source)

        for i in range(cfg.actor.actor_per_device):
            actors_key, key = jax.random.split(actors_key)
            # Create Actors
            actor = AsyncActor(
                env_factory,
                actor_device,
                params_source,
                pipeline,
                actor_fn,
                key,
                config.arch.actor,
                f"{actor_device.id}-{i}",
            )
            actors.append(actor)

    # Create Learner
    learner = AsyncLearner(
        pipeline,
        local_learner_devices,
        learner_state,
        learn,
        learner_key,
        on_params_change=[params_source.update for params_source in params_sources],
    )
    
    # Start Learner and Actors
    learner.start()
    for actor in actors:
        actor.start()
    
    try:
        # Create our stopper and wait for it to stop
        stopper = LearnerStepStopper(
            config,
            logger_manager=logger_manager
        )
        stopper.wait()
    finally:
        print(f"{Fore.RED}{Style.BRIGHT}Shutting down{Style.RESET_ALL}")
        # Try to gracefully shutdown all components
        # If not finished after 10 second exits anyway

        def graceful_shutdown():
            for actor in actors:
                actor.stop()
            for params_source in params_sources:
                params_source.stop()

            for actor in actors:
                actor.join()

            learner.stop()

            learner.join()
            pipeline.stop()
            pipeline.join()
            logger_manager.stop()
            for params_source in params_sources:
                params_source.join()
            logger_manager.join()

        graceful_thread = threading.Thread(target=graceful_shutdown, daemon=True)
        graceful_thread.start()
        graceful_thread.join(timeout=10.0)
        if graceful_thread.is_alive():
            print(f"{Fore.RED}{Style.BRIGHT}Shutdown was not graceful{Style.RESET_ALL}")

    # Setup evaluator.
    # evaluator, absolute_metric_evaluator, (trained_params, eval_keys) = evaluator_setup(
    #     eval_env=eval_env,
    #     key_e=key_e,
    #     eval_act_fn=get_distribution_act_fn(config, actor_network.apply),
    #     params=learner_state.params.actor_params,
    #     config=config,
    # )

    # Calculate number of updates per evaluation.
    # config.arch.num_updates_per_eval = config.arch.num_updates // config.arch.num_evaluation
    # steps_per_rollout = (
    #     n_devices
    #     * config.arch.num_updates_per_eval
    #     * config.system.rollout_length
    #     * config.arch.update_batch_size
    #     * config.arch.num_envs
    # )

    

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
