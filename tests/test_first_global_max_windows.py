import importlib.util
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location('summary', Path(__file__).resolve().parents[1] / 'experimental_data_analysis/pilot_summary_plots.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def trajectory(pid='a', env=0, block=0, hit=6):
    return pd.DataFrame({'id': pid, 'env': env, 'block': block, 'trial': range(1, 9),
                         'hit_global_max': [int(t >= hit) for t in range(1, 9)],
                         'search_distance': [np.nan, 2., 3., 4., 5., 99., 7., 8.],
                         'normalized_score': np.arange(1, 9) / 10,
                         'performance_group': 'low'})


def extract(df, x=3):
    return summary.extract_first_global_max_windows(df, window_size=x, participant_id_col='id')


def test_first_hit_and_alignment():
    df = pd.concat([trajectory(), trajectory(block=1, hit=5), trajectory(env=1, hit=20)])
    result = extract(df.sample(frac=1, random_state=1))
    assert len(result) == 1
    row = result.iloc[0]
    assert row.hit_block == 0 and row.hit_trial == 6 and row.eligible
    assert row.search_distance_median == 4
    assert row.search_distance_iqr == 1
    assert row.reward_median == pytest.approx(.4)
    assert row.reward_iqr == pytest.approx(.1)
    assert extract(trajectory(), 2).iloc[0].search_distance_median == 4.5
    assert extract(trajectory(), 1).iloc[0].reward_iqr == 0


def test_ineligible_first_hit_never_replaced_and_coverage():
    df = pd.concat([trajectory(hit=2), trajectory(block=1), trajectory(env=1, hit=20)])
    events = extract(df)
    assert len(events) == 1 and events.iloc[0].hit_trial == 2
    assert events.iloc[0].exclusion_reason == 'insufficient_history'
    participants, groups = summary.summarize_first_global_max_windows(events, df, participant_id_col='id')
    row = participants.iloc[0]
    assert row.n_first_hits == 1 and row.n_never_hit == 1
    assert row.n_ineligible_first_hits == 1 and row.n_eligible_environments == 0
    assert groups.iloc[0].n_contributing_participants == 0


@pytest.mark.parametrize('kind,reason', [('gap', 'nonconsecutive_trials'), ('nan', 'nonfinite_window_values'), ('block', 'insufficient_history')])
def test_invalid_windows(kind, reason):
    df = trajectory()
    if kind == 'gap':
        df = df[df.trial != 4]
    elif kind == 'nan':
        df.loc[df.trial == 4, 'normalized_score'] = np.inf
    else:
        df.loc[df.trial >= 5, 'block'] = 1
    assert extract(df).iloc[0].exclusion_reason == reason
    assert not extract(df).iloc[0].eligible
    assert not extract(trajectory(), 5).iloc[0].eligible  # initial distance missing


def test_equal_participant_weights_and_plots():
    a = trajectory()
    b1, b2 = trajectory('b'), trajectory('b', env=1)
    b1['search_distance'] = b2['search_distance'] = 10.
    never = trajectory('c', hit=20)
    never['performance_group'] = 'high'
    df = pd.concat([a, b1, b2, never])
    df['performance_group'] = pd.Categorical(df.performance_group, categories=['low', 'high'], ordered=True)
    events = extract(df)
    participants, groups = summary.summarize_first_global_max_windows(events, df, participant_id_col='id')
    low = groups.iloc[0]
    assert low.search_distance_median_median == 7  # equal participant weight, not pooled median=10
    assert low.search_distance_median_iqr == 3
    assert low.n_eligible_environments == 3 and low.n_contributing_participants == 2
    assert participants.set_index('id').loc['c', 'performance_group'] == 'high'
    figures = summary.plot_first_global_max_by_performance(events, df, participant_id_col='id', bins=2)
    values = figures['search_distance_median'][1].patches[0].get_data()
    np.testing.assert_allclose(values.values * np.diff(values.edges), [.5, .5])
    assert len(summary._viz.plot_first_global_max_participant_distributions(events, df, participant_id_col='id')) == 4
    plt.close('all')


def test_empty_singleton_and_validation():
    df = trajectory(hit=20)
    events = extract(df)
    assert events.empty
    summary.summarize_first_global_max_windows(events, df, participant_id_col='id')
    summary.plot_first_global_max_by_performance(events, df, participant_id_col='id')
    summary._viz.plot_first_global_max_participant_distributions(events, df, participant_id_col='id')
    plt.close('all')
    single = extract(trajectory())
    participants, _ = summary.summarize_first_global_max_windows(single, trajectory(), participant_id_col='id')
    assert participants.iloc[0].reward_median_iqr == 0
    for x in [0, -1, True, 1.5]:
        with pytest.raises(ValueError, match='positive integer'):
            extract(df, x)
    with pytest.raises(ValueError, match='Duplicate trials'):
        extract(pd.concat([df, df]))


def test_skewed_window_and_summary_use_median_and_linear_iqr():
    df = trajectory()
    df.loc[df.trial.isin([3, 4, 5]), 'search_distance'] = [1., 2., 100.]
    df.loc[df.trial.isin([3, 4, 5]), 'normalized_score'] = [.1, .2, 1.]
    row = extract(df).iloc[0]
    assert row.search_distance_median == 2
    assert row.search_distance_iqr == 49.5
    assert row.reward_median == .2
    assert row.reward_iqr == pytest.approx(.45)
    frames = []
    for pid, levels in [('a', [1., 2., 100.]), ('b', [4.]), ('c', [100.])]:
        for env, level in enumerate(levels):
            frame = trajectory(pid, env)
            frame['search_distance'] = level
            frames.append(frame)
    population = pd.concat(frames)
    participants, groups = summary.summarize_first_global_max_windows(
        extract(population), population, participant_id_col='id')
    assert participants.set_index('id').loc['a', 'search_distance_median_median'] == 2
    assert participants.set_index('id').loc['a', 'search_distance_median_iqr'] == 49.5
    assert groups.iloc[0].search_distance_median_median == 4
    assert groups.iloc[0].search_distance_median_iqr == 49
