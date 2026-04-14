from abc import ABC, abstractmethod
from loguru import logger
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

BANDIT_REWARDS = [1, 2, 3, 4]
NUM_TRIALS = 20000


class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        self.p = p  # true reward mean
        self.p_estimate = 0.0
        self.N = 0  # number of pulls
        self.r_estimate = 0.0

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self, algorithm):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass


# --------------------------------------#


class Visualization:

    @staticmethod
    def plot1(eg_bandits, ts_bandits, eg_rewards, ts_rewards):
        # visualize the performance of each bandit: linear and log
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("bandit performance over time", fontsize=16, fontweight="bold")

        # epsilon-greedy linear scale
        ax = axes[0, 0]
        for i, b in enumerate(eg_bandits):
            if b._history:
                cumulative = np.cumsum(b._history)
                counts = np.arange(1, len(b._history) + 1)
                ax.plot(counts, cumulative / counts, label=f"bandit {b.p}")
        ax.set_title("epsilon-greedy (linear)")
        ax.set_xlabel("trials")
        ax.set_ylabel("average reward")
        ax.legend()

        # epsilon-greedy log scale
        ax = axes[0, 1]
        for i, b in enumerate(eg_bandits):
            if b._history:
                cumulative = np.cumsum(b._history)
                counts = np.arange(1, len(b._history) + 1)
                ax.plot(counts, cumulative / counts, label=f"bandit {b.p}")
        ax.set_xscale("log")
        ax.set_title("epsilon-greedy (Log)")
        ax.set_xlabel("trials (log)")
        ax.set_ylabel("average reward")
        ax.legend()

        # thompson sampling linear scale
        ax = axes[1, 0]
        for i, b in enumerate(ts_bandits):
            if b._history:
                cumulative = np.cumsum(b._history)
                counts = np.arange(1, len(b._history) + 1)
                ax.plot(counts, cumulative / counts, label=f"bandit {b.p}")
        ax.set_title("thompson sampling (linear)")
        ax.set_xlabel("trials")
        ax.set_ylabel("average reward")
        ax.legend()

        # thompson sampling lLog scale
        ax = axes[1, 1]
        for i, b in enumerate(ts_bandits):
            if b._history:
                cumulative = np.cumsum(b._history)
                counts = np.arange(1, len(b._history) + 1)
                ax.plot(counts, cumulative / counts, label=f"bandit {b.p}")
        ax.set_xscale("log")
        ax.set_title("thompson sampling (log)")
        ax.set_xlabel("trials (log)")
        ax.set_ylabel("average reward")
        ax.legend()

        plt.tight_layout()
        plt.savefig("plot1_bandit_performance.png", dpi=150)
        plt.show()
        logger.info("saved plot1_bandit_performance.png")

    @staticmethod
    def plot2(eg_rewards, ts_rewards):
        # compare e-greedy and thompson sampling cumulative rewards and regrets
        best_reward = max(BANDIT_REWARDS)
        eg_cum_reward = np.cumsum(eg_rewards)
        ts_cum_reward = np.cumsum(ts_rewards)
        eg_cum_regret = np.cumsum([best_reward - r for r in eg_rewards])
        ts_cum_regret = np.cumsum([best_reward - r for r in ts_rewards])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("epsilon-greedy vs thompson sampling", fontsize=16, fontweight="bold")

        # cumulative reward
        axes[0].plot(eg_cum_reward, label="epsilon-greedy")
        axes[0].plot(ts_cum_reward, label="thompson sampling")
        axes[0].set_title("cumulative reward")
        axes[0].set_xlabel("trials")
        axes[0].set_ylabel("cumulative reward")
        axes[0].legend()

        # cumulative regret
        axes[1].plot(eg_cum_regret, label="epsilon-greedy")
        axes[1].plot(ts_cum_regret, label="thompson sampling")
        axes[1].set_title("cumulative regret")
        axes[1].set_xlabel("trials")
        axes[1].set_ylabel("cumulative regret")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig("plot2_comparison.png", dpi=150)
        plt.show()
        logger.info("saved plot2_comparison.png")


# --------------------------------------#


class EpsilonGreedy(Bandit):

    def __init__(self, p):
        super().__init__(p)
        self.p_estimate = 0.0
        self.N = 0
        self._history = []  # per-bandit reward history

    def __repr__(self):
        return f"epsilon-greedy(p={self.p}, estimate={self.p_estimate:.4f}, N={self.N})"

    def pull(self):
        return np.random.randn() + self.p

    def update(self, x):
        self.N += 1
        self.p_estimate = ((self.N - 1) * self.p_estimate + x) / self.N
        self._history.append(x)

    def experiment(self, bandits, num_trials=NUM_TRIALS):
        rewards = []
        optimal = np.argmax([b.p for b in bandits])

        for t in range(1, num_trials + 1):
            epsilon = 1 / t  # epsilon decay
            if np.random.random() < epsilon:
                j = np.random.randint(len(bandits))
            else:
                j = np.argmax([b.p_estimate for b in bandits])

            x = bandits[j].pull()
            bandits[j].update(x)
            rewards.append(x)

        return rewards

    def report(self, algorithm="epsilon-greedy"):
        pass  # handled by standalone report()


# --------------------------------------#


class ThompsonSampling(Bandit):

    def __init__(self, p):
        super().__init__(p)
        self.mu = 0.0  # posterior mean
        self.tau = 1.0  # posterior precision
        self.N = 0
        self._sum = 0.0
        self._history = []

    def __repr__(self):
        return f"thompson-sampling(p={self.p}, mu={self.mu:.4f}, tau={self.tau:.4f}, N={self.N})"

    def pull(self):
        return np.random.randn() + self.p

    def update(self, x):
        self.N += 1
        self._sum += x
        # gaussian posterior update with known precision tau0=1
        self.tau = 1 + self.N  # prior precision + n * known precision
        self.mu = self._sum / self.tau
        self._history.append(x)

    def experiment(self, bandits, num_trials=NUM_TRIALS):
        rewards = []

        for _ in range(num_trials):
            # sample from posterior for each bandit
            samples = [np.random.randn() / np.sqrt(b.tau) + b.mu for b in bandits]
            j = np.argmax(samples)

            x = bandits[j].pull()
            bandits[j].update(x)
            rewards.append(x)

        return rewards

    def report(self, algorithm="thompson-sampling"):
        pass  # handled by standalone report()


# --------------------------------------#


# save CSV and log cumulative reward/regret
def report(rewards, bandits, algorithm):
    best_reward = max(BANDIT_REWARDS)
    cumulative_reward = np.sum(rewards)
    cumulative_regret = best_reward * len(rewards) - cumulative_reward

    logger.info(f"[{algorithm}] cumulative reward: {cumulative_reward:.2f}")
    logger.info(f"[{algorithm}] cumulative regret: {cumulative_regret:.2f}")

    # build csv data
    records = []
    for i, b in enumerate(bandits):
        for r in b._history:
            records.append({"bandit": b.p, "reward": r, "algorithm": algorithm})

    df = pd.DataFrame(records)
    filename = f"{algorithm}_results.csv"
    df.to_csv(filename, index=False)
    logger.info(f"[{algorithm}] saved trial data to {filename}")

    return df


def comparison():
    # epsilon-greed
    eg_bandits = [EpsilonGreedy(p) for p in BANDIT_REWARDS]
    eg = eg_bandits[0]
    eg_rewards = eg.experiment(eg_bandits, NUM_TRIALS)
    eg_df = report(eg_rewards, eg_bandits, "epsilon-greedy")

    # thompson sampling
    ts_bandits = [ThompsonSampling(p) for p in BANDIT_REWARDS]
    ts = ts_bandits[0]
    ts_rewards = ts.experiment(ts_bandits, NUM_TRIALS)
    ts_df = report(ts_rewards, ts_bandits, "thompson-sampling")

    # visualization
    Visualization.plot1(eg_bandits, ts_bandits, eg_rewards, ts_rewards)
    Visualization.plot2(eg_rewards, ts_rewards)


if __name__ == "__main__":
    comparison()
