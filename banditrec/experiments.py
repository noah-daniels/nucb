from banditrec.agents import HindsightAgent, RandomAgent
from banditrec.simulate import SimulatorEC, SimulatorRP, SimulatorSW
from banditrec.datasets import (
    InteractionDataset,
    import_lastfm,
    import_delicious,
    import_lastfm2,
    import_delicious2,
)
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math
from scipy.signal import savgol_filter


def load_datasets(base_path):
    R_lastfm, _ = import_lastfm(
        f"{base_path}/hetrec2011-lastfm-2k/user_artists.dat", 1000
    )
    R_delicious, _ = import_delicious(f"{base_path}/hetrec2011-delicious-2k", 1000)
    R_lastfmx, features_lastfm = import_lastfm2(
        f"{base_path}/hetrec2011-lastfm-2k", d=25
    )
    R_deliciousx, features_delicious = import_delicious2(
        f"{base_path}/hetrec2011-delicious-2k", d=25
    )

    return {
        "lastfm": InteractionDataset(R_lastfm, "lastfm", 200000),
        "delicious": InteractionDataset(R_delicious, "delicious", 200000),
        "lastfmx": InteractionDataset(R_lastfmx, "lastfmx", 50000).add_item_features(
            features_lastfm
        ),
        "deliciousx": InteractionDataset(
            R_deliciousx, "deliciousx", 50000
        ).add_item_features(features_delicious),
    }


class Experiment:
    def __init__(self, output_path):
        self.output_path = output_path

        # ensure output path exists
        Path(output_path).mkdir(parents=True, exist_ok=True)

        self.run_count = 1
        self.pool_size = 25
        self.datasets = []

    def set_run_count(self, value):
        self.run_count = value
        return self

    def run_multiple(self, datasets, variants, agent, name=None):
        for dataset in datasets:
            for variant in variants:
                self.run(dataset, variant, agent, name)

    def run(self, dataset, variant, agent, name=None):
        if variant == "EC":
            simulator = SimulatorEC(dataset)
        elif variant == "RP":
            simulator = SimulatorRP(dataset, self.pool_size)
        elif variant == "SW":
            simulator = SimulatorSW(dataset, self.pool_size)
        else:
            assert False

        if name is None:
            name = agent if isinstance(agent, str) else agent.__class__.__name__

        simulator.set_run_count(self.run_count)

        if agent == "Random":
            agent = RandomAgent()
        elif agent == "Hindsight":
            agent = HindsightAgent(ranked_items=dataset.ranked_items)

        if dataset.item_features is not None:
            agent.set_features(dataset.item_features)

        result = simulator.run(agent)
        np.save(
            f"{self.output_path}/{dataset.name}{variant}_{name}.npy", result.rewards
        )


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    result = np.zeros(len(y)) + np.nan
    result[box_pts // 2 : -box_pts // 2] = np.convolve(y, box, mode="same")[
        box_pts // 2 : -box_pts // 2
    ]
    return result


class ExperimentVisualizer:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

        # ensure output path exists
        Path(output_path).mkdir(parents=True, exist_ok=True)

    def new_plot(self, prefix, time_horizon, bucket_size=None):
        self.prefix = prefix
        self.time_horizon = time_horizon
        self.bucket_size = (
            bucket_size if bucket_size is not None else time_horizon // 500
        )
        self.results = []

    def add_result(self, name, options=None):
        if self.prefix is None:
            path = f"{self.input_path}/{name}.npy"
        else:
            path = f"{self.input_path}/{self.prefix}_{name}.npy"

        try:
            rewards = np.load(path)
            self.results.append((name, rewards, options))
        except:
            print("Could not find", path)

    def render_all(self):
        self.create_plot("rewards")
        self.create_plot("ctrs")
        self.create_plot("regrets")

    def create_plot(self, variant, prefix=None):
        fig, ax = plt.subplots(
            figsize=(5, 4),
            dpi=200,
        )

        for name, rewards, options in self.results:
            x = self.bucket_size * np.arange(1, 1 + len(rewards))
            if options is None:
                options = {}
            if "label" not in options:
                options["label"] = name

            if variant == "rewards":
                values = np.cumsum(rewards)
            elif variant == "ctrs":
                values = np.cumsum(rewards) / x
            elif variant == "regrets":
                values = np.cumsum(self.bucket_size - rewards) / np.cumsum(
                    self.bucket_size - self.results[0][1]
                )
            elif variant == "gradient":
                values = np.gradient(
                    savgol_filter(np.cumsum(rewards) / self.bucket_size, 21, 3)
                )
            else:
                assert False

            # ax.plot(x[2:], values[2:], **options)
            ax.plot(x, values, **options)

        ax.set_xticks(np.arange(0, self.time_horizon + 1, self.time_horizon // 5))

        # plt.xticks(10000 * np.linspace(0, 7, 8))
        plt.xlabel("Rounds")
        plt.grid()
        plt.legend()

        if prefix is None:
            prefix = ""
        else:
            prefix = f"{prefix}_"

        fig.savefig(f"{self.output_path}/{prefix}{variant}_{self.prefix}.png")
        plt.close()


class ExperimentVisualizer2(ExperimentVisualizer):
    def new_plot(self, datasets, time_horizon, bucket_size=None, contextual=False):
        self.datasets = datasets
        self.time_horizon = time_horizon
        self.bucket_size = (
            bucket_size if bucket_size is not None else time_horizon // 500
        )
        self.results = []
        self.contextual = contextual

    def add_result(self, name, options=None):
        self.results.append((name, options))

    def create_plot(self, variant, imgname, dataset_names=None):
        fig, axes = plt.subplots(
            *np.array(self.datasets).shape,
            figsize=(6, 6),
            dpi=200,
            layout="constrained",
            sharex=False,
        )

        handles = []
        for i, row in enumerate(self.datasets):
            for j, d in enumerate(row):
                ax = axes[i, j]

                if isinstance(variant, list):
                    v = variant[i]
                else:
                    v = variant

                b = self.bucket_size
                if self.contextual and i == 1:
                    b = 100_000 // 500

                for name, options in self.results:
                    try:
                        path = f"{self.input_path}/{d}_{name}.npy"
                        rewards = np.load(path)
                        if name == "Random":
                            random_rewards = rewards
                    except:
                        continue

                    x = b * np.arange(1, 1 + len(rewards))
                    if options is None:
                        options = {}
                    if "label" not in options:
                        options["label"] = name

                    if v == "rewards":
                        values = np.cumsum(rewards)

                    elif v == "ctrs":
                        values = np.cumsum(rewards) / x
                    elif v == "regrets":
                        values = np.cumsum(b - rewards) / np.cumsum(
                            self.bucket_size - random_rewards
                        )
                        if j == 0:
                            ax.set_ylabel("Relative cumulative regret")
                    elif v == "gradient":
                        values = np.gradient(
                            savgol_filter(np.cumsum(rewards) / b, 21, 3)
                        )
                    else:
                        assert False

                    (l,) = ax.plot(x, values, **options)
                    if i == 0 and j == 0:
                        handles.append(l)

                if i == len(self.datasets) - 1:
                    ax.set_xlabel("Rounds")

                if j == 1:
                    ax.yaxis.tick_right()

                if j == 0:
                    if v == "rewards":
                        ax.set_ylabel("Cumulative reward")
                    elif v == "regregts":
                        ax.set_ylabel("Relative cumulative regret")

                if dataset_names is None:
                    ax.set_title(d)
                else:
                    ax.set_title(dataset_names[i][j])
                ax.grid(True)

                ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
                ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

        fig.legend(
            loc="outside lower center",
            handles=handles,
            ncols=math.ceil(len(self.results) / 2),
        )

        fig.get_layout_engine().set(hspace=0.1)

        fig.savefig(f"{self.output_path}/{imgname}.png")
        plt.close()
