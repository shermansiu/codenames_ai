import sys
import gym
import pathlib
import typing as tp
from scipy.special import softmax
import pandas as pd
import os
from gridtest_env import *
from plot_utils import *
import stable_baselines3
from stable_baselines3.common import env_checker
from stable_baselines3 import DQN, SAC, PPO, A2C, HerReplayBuffer
import matplotlib.pyplot as plt
from stable_baselines3 import TD3
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common import env_checker

env_checker.check_env(WhackAMoleMultiBinaryEnv(length=2))

from stable_baselines3.common.logger import configure
import pathlib

EXPERIMENT_NAME = "whackamole_multibinary_l2"
EXPERIMENT_DISPLAY_NAME = "Whack-A-Mole - Multibinary - Length 2"
CODENAMES_DIR = pathlib.Path(__file__).resolve().parent.parent
tmp_path = str(CODENAMES_DIR.joinpath(f"logs/{EXPERIMENT_NAME}/sb3_log/_"))[:-1]
# set up logger
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])


def make_env(env_name):
    log_dir = pathlib.Path(tmp_path).joinpath(env_name)
    log_dir.mkdir(parents=True, exist_ok=True)
    return Monitor(WhackAMoleMultiBinaryEnv(length=2), str(log_dir))


log_freq = 25000
num_train_steps = 1_000_000


n_sampled_goal = 4
models = dict(
    # sac = SAC('MultiInputPolicy', make_env("sac"), verbose=1, target_update_interval=25),
    # sac_her = SAC(
    #     'MultiInputPolicy',
    #     make_env("sac_her"),
    #     verbose=1, target_update_interval=25,
    #     replay_buffer_class=HerReplayBuffer,
    #     replay_buffer_kwargs=dict(
    #       n_sampled_goal=n_sampled_goal,
    #       goal_selection_strategy="future",
    #       # IMPORTANT: because the env is not wrapped with a TimeLimit wrapper
    #       # we have to manually specify the max number of steps per episode
    #       max_episode_length=25,
    #       online_sampling=True,
    #     ),
    #     buffer_size=int(1e6)
    # ),
    ppo = PPO('MlpPolicy', make_env("ppo"), verbose=1, n_steps=99, n_epochs=80),
    a2c = A2C('MlpPolicy', make_env("a2c"), verbose=1, n_steps=99)
)

from tqdm.auto import tqdm

class ProgressBarCallback(BaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """
    def __init__(self, pbar):
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar

    def _on_step(self):
        # Update the progress bar:
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)

# this callback uses the 'with' block, allowing for correct initialisation and destruction
class ProgressBarManager:
    def __init__(self, total_timesteps): # init object with total timesteps
        self.pbar = None
        self.total_timesteps = total_timesteps
        
    def __enter__(self): # create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.total_timesteps)
            
        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb): # close the callback
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()
        
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            try:
                x, y = ts2xy(load_results(self.log_dir), 'timesteps')
                if len(x) > 0:
                    # Mean training reward over the last 100 episodes
                    mean_reward = np.mean(y[-100:])
                    if self.verbose > 0:
                        print(f"Num timesteps: {self.num_timesteps}")
                        print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                    # New best model, you could save the agent here
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        # Example for saving best model
                        if self.verbose > 0:
                            print(f"Saving new best model to {self.save_path}")
                        self.model.save(self.save_path)
            except Exception:
                pass
        return True


# We don't support discrete mode right now
tmp_path_discrete = tmp_path
def get_path_for(env_name, is_discrete=False):
    if is_discrete:
        return str(pathlib.Path(tmp_path_discrete).joinpath(env_name))
    return str(pathlib.Path(tmp_path).joinpath(env_name))


class Run:
    def __init__(self, name, model, total_timesteps=1, mode="self_sim", num_eval_episodes=10, should_render=False, log_interval=500):
        assert mode in {"self_sim", "discrete"}
        print("Model:", name)
        self.log_dir = get_path_for(name, mode == "discrete")
        self.save_callback = SaveOnBestTrainingRewardCallback(check_freq=log_freq, log_dir=self.log_dir)
        self.logger = configure(get_path_for(name + "_logger", mode == "discrete"), ["stdout", "csv", "tensorboard"])
        print("Start training")
        model.set_logger(self.logger)
        with ProgressBarManager(total_timesteps) as progress_callback: # this the garanties that the tqdm progress bar closes correctly
            model.learn(total_timesteps=total_timesteps, log_interval=log_interval, callback=CallbackList([progress_callback, self.save_callback]))
        # Save the agent
        print("Saving...")
        model.save(f"{EXPERIMENT_NAME}_{name}_v0.0.1_end.chkpt.zip")
        # Evaluate the trained agent
        print("Evaluating...")
        if mode == "self_sim":
            self.eval_monitor_trained = make_env(name + "_eval")
        else:
            self.eval_monitor_trained = make_env_discrete(name + "_eval")
        mean_reward, std_reward = evaluate_policy(model, self.eval_monitor_trained, n_eval_episodes=num_eval_episodes, deterministic=True, render=should_render)
        print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
        self.print_monitor_stat("Lengths", self.eval_monitor_trained.episode_lengths)
        self.print_monitor_stat("Returns", self.eval_monitor_trained.episode_returns)

    def print_monitor_stat(self, stat_name, stat):
        print(stat_name, stat, np.mean(stat), np.std(stat))

        
runs = {k: Run(k, m, num_train_steps, should_render=False) for k, m in models.items()}


class PlotHack:
    def visualise_overall_agent_results(self, agent_results, agent_name, show_mean_and_std_range=False, show_each_run=False,
                                        color=None, ax=None, title=None, y_limits=None):
        """Visualises the results for one agent"""
        assert isinstance(agent_results, list), "agent_results must be a list of lists, 1 set of results per list"
        assert isinstance(agent_results[0], list), "agent_results must be a list of lists, 1 set of results per list"
        assert bool(show_mean_and_std_range) ^ bool(show_each_run), "either show_mean_and_std_range or show_each_run must be true"
        if not ax: ax = plt.gca()
        if not color: color =  self.agent_to_color_group[agent_name]
        if show_mean_and_std_range:
            mean_minus_x_std, mean_results, mean_plus_x_std = self.get_mean_and_standard_deviation_difference_results(agent_results)
            x_vals = list(range(len(mean_results)))
            ax.plot(x_vals, mean_results, label=agent_name, color=color)
            ax.plot(x_vals, mean_plus_x_std, color=color, alpha=0.1)
            ax.plot(x_vals, mean_minus_x_std, color=color, alpha=0.1)
            ax.fill_between(x_vals, y1=mean_minus_x_std, y2=mean_plus_x_std, alpha=0.1, color=color)
        else:
            for ix, result in enumerate(agent_results):
                x_vals = list(range(len(agent_results[0])))
                plt.plot(x_vals, result, label=agent_name + "_{}".format(ix+1), color=color)
                color = self.get_next_color()

        ax.set_facecolor('xkcd:white')

        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.05,
                         box.width, box.height * 0.95])

        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  fancybox=True, shadow=True, ncol=3)

        if not title: title = self.environment_name

        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.set_ylabel('Rolling Episode Scores')
        ax.set_xlabel('Episode Number')
        self.hide_spines(ax, ['right', 'top'])
        ax.set_xlim([0, x_vals[-1]])

        if y_limits is None: y_min, y_max = self.get_y_limits(agent_results)
        else: y_min, y_max = y_limits

        ax.set_ylim([y_min, y_max])

        if self.config.show_solution_score:
            self.draw_horizontal_line_with_label(ax, y_value=self.config.environment.get_score_to_win(), x_min=0,
                                        x_max=self.config.num_episodes_to_run * 1.02, label="Target \n score")


plot_gridtest_experiment(EXPERIMENT_NAME, EXPERIMENT_DISPLAY_NAME)
