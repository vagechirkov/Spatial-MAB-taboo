import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "summary", Path(__file__).resolve().parents[1] / "experimental_data_analysis/pilot_summary_plots.py"
)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def trajectory(choices, pid="a", env=0, block=0, group="low"):
    return pd.DataFrame({"id": pid, "env": env, "block": block,
                         "trial": np.arange(len(choices)), "choice_x": choices,
                         "choice_y": 0, "performance_group": group})


def summarize(df):
    return summary.summarize_choice_reselection(df, participant_id_col="id")


def test_repeats_sorting_resets_and_pooled_counts():
    df = pd.concat([trajectory([0, 1, 0, 0]), trajectory([0, 0], block=1),
                    trajectory([0], env=1), trajectory([0], pid="b")], ignore_index=True)
    original = df.copy(deep=True)
    trials, participants, _ = summarize(df.sample(frac=1, random_state=10))
    a = participants.set_index(["id", "env"]).loc[("a", 0)]
    assert a.n_reselections == 3 and a.n_turns == 6
    assert a.reselection_fraction == .5  # pooled counts across blocks
    assert not trials.loc[trials.trial.eq(0), "is_reselection"].any()
    first = trials.loc[trials.id.eq("a") & trials.env.eq(0) & trials.block.eq(0)]
    assert first.is_reselection.tolist() == [False, False, True, True]
    pd.testing.assert_frame_equal(df, original)
    assert len(participants) == 3  # missing b/env1 is not synthesized
    _, pooled, _ = summarize(pd.concat([trajectory([0, 0, 0, 0]), trajectory([0], block=1)]))
    assert pooled.iloc[0].reselection_fraction == .6  # not mean(.75, 0)


@pytest.mark.parametrize("choices,expected", [([0], 0), ([0, 1, 2], 0), ([0, 0, 0], 2 / 3)])
def test_simple_sequences(choices, expected):
    _, participants, _ = summarize(trajectory(choices))
    assert participants.iloc[0].reselection_fraction == expected


def test_equal_weights_quantiles_and_plots():
    df = pd.concat([trajectory([0], pid="a"), trajectory([0, 0], pid="b"),
                    trajectory([0] * 10, pid="c"), trajectory([0, 1], pid="d", group="high")])
    df["performance_group"] = pd.Categorical(df.performance_group, categories=["low", "high", "unused"], ordered=True)
    _, participants, groups = summarize(df)
    low = groups.loc[groups.performance_group.eq("low")].iloc[0]
    assert low.n_participants == 3
    assert low["median"] == .5
    assert low.q25 == .25 and low.q75 == .7
    assert low.iqr == pytest.approx(.45)
    assert participants.set_index("id").loc["d", "performance_group"] == "high"
    fig, axes = summary._viz.plot_choice_reselection_by_participant(participants, participant_id_col="id")
    assert len(axes.flat[0].patches) == 1
    assert axes.flat[0].get_ylim() == (0, 1)
    fig, axes = summary.plot_choice_reselection_by_performance(participants, groups)
    assert [ax.get_title() for ax in axes.flat] == ["low", "high", "unused"]
    assert all(ax.get_ylim() == (0, 1) for ax in axes.flat)
    plt.close("all")


@pytest.mark.parametrize("column,value", [("id", None), ("id", np.inf), ("env", np.nan),
    ("block", np.inf), ("trial", "bad"), ("choice_x", np.inf), ("choice_y", np.nan),
    ("performance_group", None)])
def test_invalid_rows(column, value):
    df = trajectory([0, 1])
    df[column] = value
    with pytest.raises(ValueError):
        summarize(df)


def test_duplicates_membership_missing_columns_and_empty():
    df = trajectory([0, 1])
    with pytest.raises(ValueError, match="Duplicate"):
        summarize(pd.concat([df, df]))
    with pytest.raises(ValueError, match="single performance group"):
        summarize(pd.concat([df, trajectory([0], block=1, group="high")]))
    with pytest.raises(ValueError, match="Missing columns"):
        summarize(df.drop(columns="choice_x"))
    trials, participants, groups = summarize(df.iloc[:0])
    assert trials.empty and participants.empty and groups.empty
    assert "is_reselection" in trials and "reselection_fraction" in participants and "iqr" in groups
    summary._viz.plot_choice_reselection_by_participant(participants, participant_id_col="id")
    summary.plot_choice_reselection_by_performance(participants, groups)
    plt.close("all")
