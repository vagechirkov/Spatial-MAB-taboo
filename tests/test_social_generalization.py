import mesa
import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.gaussian_process.kernels import RBF

from abm.agent import gp_base_generalization, social_generalization
from abm.model import SocialGPModel, as_batch_fixed
from abm.run_slurm_jobs import build_parser


def test_social_generalization_matches_manual_gp_posterior():
    X_private = np.array([[0.0, 0.0], [1.0, 0.0]])
    y_private = np.array([0.2, 0.5])
    X_social = [np.array([[0.0, 1.0]])]
    y_social = [np.array([0.9])]
    X_predict = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    length_scale = 1.3
    base_noise = 0.05
    sigma_social = 0.7
    beta = 0.4

    actual = social_generalization(
        X_private,
        y_private,
        X_social,
        y_social,
        X_predict,
        length_scale,
        base_noise,
        sigma_social,
        beta,
        random_state=0,
    )

    X_observed = np.vstack((X_private, X_social[0]))
    y_observed = np.concatenate((y_private, y_social[0]))
    kernel = RBF(length_scale=length_scale)
    noise = np.array([base_noise, base_noise, base_noise + sigma_social])
    K = kernel(X_observed, X_observed) + np.diag(noise)
    K_star = kernel(X_predict, X_observed)
    mean = K_star @ np.linalg.solve(K, y_observed)
    covariance = kernel(X_predict, X_predict) - K_star @ np.linalg.solve(K, K_star.T)
    expected = mean + beta * np.sqrt(np.clip(np.diag(covariance), 0.0, None))

    assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_larger_social_noise_discounts_social_reward():
    kwargs = dict(
        X_obs_private=np.array([[0.0, 0.0]]),
        y_obs_private=np.array([0.0]),
        X_obs_social=[np.array([[0.0, 0.0]])],
        y_obs_social=[np.array([1.0])],
        X_predict=np.array([[0.0, 0.0]]),
        length_scale=1.0,
        observation_noise=0.01,
        beta=0.0,
        random_state=0,
    )

    low_noise_value = social_generalization(sigma_social=0.0, **kwargs)[0]
    high_noise_value = social_generalization(sigma_social=100.0, **kwargs)[0]

    assert high_noise_value < low_noise_value
    assert high_noise_value == pytest.approx(0.0, abs=1e-3)


def test_no_social_history_matches_private_gp():
    X_private = np.array([[0.0, 0.0], [1.0, 1.0]])
    y_private = np.array([0.2, 0.7])
    X_predict = np.array([[0.0, 0.0], [0.0, 1.0]])

    expected_mean, expected_std = gp_base_generalization(
        X_private,
        y_private,
        X_predict,
        RBF(length_scale=1.2),
        np.full(2, 0.03),
        rng=0,
    )
    actual = social_generalization(
        X_private,
        y_private,
        [np.empty((0, 2))],
        [np.array([])],
        X_predict,
        length_scale=1.2,
        observation_noise=0.03,
        sigma_social=12.55,
        beta=0.6,
        random_state=0,
    )

    assert_allclose(actual, expected_mean + 0.6 * expected_std)


def test_model_mode_validation_and_sigma_normalization():
    model = SocialGPModel(
        n=2,
        grid_size=4,
        social_information_mode="social_generalization",
        sigma_social=as_batch_fixed([1.0, 2.0]),
        env_seed=1,
        run_seed=2,
    )

    assert model.social_information_mode == "social_generalization"
    assert model.agent_hyperparameters["sigma_social"] == [1.0, 2.0]
    assert [agent.sigma_social for agent in model.agents] == [1.0, 2.0]

    with pytest.raises(ValueError, match="social_information_mode"):
        SocialGPModel(n=1, grid_size=3, social_information_mode="unknown")
    for invalid_sigma in (-1.0, np.inf, np.nan):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            SocialGPModel(n=1, grid_size=3, sigma_social=invalid_sigma)


def test_social_history_excludes_unfinished_round():
    model = SocialGPModel(n=2, grid_size=4, env_seed=1, run_seed=2)
    focal, neighbour = list(model.agents)
    neighbour.X_observations[:] = [(0, 0), (1, 1)]
    neighbour.y_observations[:] = [0.1, 0.9]
    model.steps = 2

    X_social, y_social = focal._gather_social_info()

    assert X_social[0].tolist() == [[0, 0]]
    assert y_social[0].ravel().tolist() == [0.1]


@pytest.mark.parametrize("mode", ["value_shaping", "social_generalization"])
def test_mesa_batch_run_smoke(mode):
    results = mesa.batch_run(
        SocialGPModel,
        parameters={
            "n": 2,
            "grid_size": 4,
            "social_information_mode": mode,
            "sigma_social": 1.0,
            "env_seed": 1,
            "run_seed": 2,
        },
        rng=[None],
        max_steps=3,
        display_progress=False,
        number_processes=1,
    )

    assert results
    assert {"RunId", "Step", "AgentID", "reward"}.issubset(results[0])


def test_cli_accepts_social_generalization_parameters():
    args = build_parser().parse_args(
        [
            "--social-information-mode",
            "social_generalization",
            "--sigma-social",
            "12.55",
            "--sigma-social-by-agent",
            "12.55,2.0",
        ]
    )

    assert args.social_information_mode == "social_generalization"
    assert args.sigma_social == 12.55
    assert args.sigma_social_by_agent == "12.55,2.0"
