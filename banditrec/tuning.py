import math

from banditrec.agents import HindsightAgent, RandomAgent
import numpy as np
from sklearn.model_selection import ParameterGrid
from tqdm.auto import tqdm


def tune(simulator, agent_class, parameter_grid, horizon, filename):

    results = np.empty(len(parameter_grid))

    tq = tqdm(parameter_grid, postfix=agent_class.__name__)
    for i, p in enumerate(tq):
        agent = agent_class(**p)
        if simulator.dataset.item_features is not None:
            agent.set_features(simulator.dataset.item_features)
        results[i] = simulator.run(agent, horizon).rewards.sum()

    idx = np.argsort(-results)
    elapsed = tq.format_dict["elapsed"]

    with open(filename, "a", encoding="utf-8") as file:
        file.write(f"{'-':->100}\n")
        file.write(f"{simulator.label} (N={simulator.run_count}, horizon={horizon})\n")
        file.write(
            f"Running {len(parameter_grid)} parameter combinations for {agent_class.__name__} took {tq.format_interval(elapsed)}\n\n"
        )
        for i in idx:
            file.write(
                f"{results[i]:>10.0f} - {agent_class(**parameter_grid[i]).label}\n"
            )
        file.write("\n")

    return parameter_grid[idx[0]]
