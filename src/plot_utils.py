import pathlib
from stable_baselines3.common import results_plotter
import matplotlib.pyplot as plt


CODENAMES_AI_PATH = pathlib.Path().resolve().parent


def plot_gridtest_experiment(expt_name, expt_display_name, ylim=None, should_save=True, algorithms=None):
    logs_path = CODENAMES_AI_PATH.joinpath(f"logs/{expt_name}/sb3_log")
    if algorithms is None:
        algorithms = ["PPO", "A2C"]
    algo_paths = [str(logs_path.joinpath(algo.lower())) for algo in algorithms]

    results_plotter.plot_results(
        algo_paths,
        None, results_plotter.X_EPISODES, expt_display_name
    )
    plot_labels = flatten([[algo, f"avg({algo})"] for algo in algorithms])
    plt.legend(loc="best", labels=plot_labels)
    plt.grid()
    if ylim is not None:
        plt.ylim(ylim)
    if should_save:
        plt.savefig(f"{expt_name}_sb3.png")