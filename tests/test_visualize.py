import pytest
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from swiss_cheese import MNAR, MAR, MCAR
import os


@pytest.fixture
def data():
    rng = np.random.default_rng(42)
    data = rng.random((5000, 100))
    data.sort()
    return data


def plot(data, title):
    ax = sns.heatmap(data, cmap="viridis", annot=False)
    plt.title("")
# Remove ticks and tick labels
    ax.set_xticks([])
    ax.set_yticks([])

# Remove axis labels
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.savefig(title)
    plt.close()


@pytest.mark.parametrize("mean", [0.25, 0.5, 0.75])
@pytest.mark.parametrize("variance", [0.01, 0.1, 0.2])
@pytest.mark.parametrize("mode", ["gm", "min", "max"])
def test_mnar(data, mean: float, variance: float, mode: str):
    data = MNAR(mean=mean, variance=variance, mode=mode)(data, 0.3)
    # data[np.isnan(data)] = 0
    plot(data, f"figures/mnar_{mode}_{mean}_{variance}.png")


@pytest.mark.parametrize("mean", [0.25, 0.5, 0.75])
@pytest.mark.parametrize("variance", [0.01, 0.1, 0.2])
@pytest.mark.parametrize("mode", ["gm", "min", "max"])
def test_mar(data, mean: float, variance: float, mode: str):
    data = MAR(mean=mean, variance=variance, mode=mode)(data, 0.3)
    # data[np.isnan(data)] = 0
    plot(data, f"figures/mar_{mode}_{mean}_{variance}.png")


def test_mcar(data):
    data = MCAR()(data, 0.3)
    # data[np.isnan(data)] = 0
    plot(data, f"figures/mcar.png")


def test_block(data):
    data = MNAR(mode="block")(data, 0.3)
    # data[np.isnan(data)] = 0
    plot(data, f"figures/block.png")


def test_blob(data):
    data = MNAR(mode="blob")(data, 0.3)
    # data[np.isnan(data)] = 0
    plot(data, f"figures/blob.png")
