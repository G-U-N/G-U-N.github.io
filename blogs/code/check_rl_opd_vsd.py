"""Reproduce the gradient checks in ../rl-opd-vsd.html.

Run: python3 blogs/code/check_rl_opd_vsd.py
Requires NumPy. Finite sums and Gauss-Hermite quadrature avoid sampling error.
These checks illustrate the identities; they do not replace their assumptions.
"""

import numpy as np
from numpy.testing import assert_allclose


def finite_difference(function, parameters, step=1e-5):
    parameters = np.asarray(parameters, dtype=float)
    basis = np.eye(len(parameters)) * step
    return np.array([
        (function(parameters + offset) - function(parameters - offset))
        / (2 * step)
        for offset in basis
    ])


def normal_quadrature():
    nodes, weights = np.polynomial.hermite.hermgauss(48)
    return np.sqrt(2) * nodes, weights / np.sqrt(np.pi)


def check_sequence_prefix_gradient():
    # Outcomes are (A, 0), (A, 1), (B, 0), (B, 1).
    branch_b = np.array([0., 0., 1., 1.])
    teacher = np.array([.25, .25, .05, .45])
    branch_kl = .5 * np.log(25 / 9)

    def student(eta):
        u = 1 / (1 + np.exp(-eta))
        return np.array([(1 - u) / 2] * 2 + [u / 2] * 2), u

    def sequence_kl(parameters):
        q, _ = student(parameters[0])
        return np.sum(q * np.log(q / teacher))

    for eta in [-.8, 0., .9]:
        q, u = student(eta)
        log_ratio = np.log(q / teacher)
        parameter_score = branch_b - u
        full_gradient = np.sum(q * log_ratio * parameter_score)
        # Only the first-step student probability depends on eta.
        frozen_prefix_gradient = u * (1 - u) * np.log(u / (1 - u))
        prefix_derivative = u * (1 - u) * branch_kl
        first_step_log_ratio = np.log(np.where(branch_b, u, 1 - u) / .5)
        immediate_only = np.sum(q * first_step_log_ratio * parameter_score)
        chain_rule_value = (
            u * np.log(2 * u) + (1 - u) * np.log(2 * (1 - u))
            + u * branch_kl
        )
        assert_allclose(sequence_kl([eta]), chain_rule_value, atol=1e-12)
        assert_allclose(full_gradient, finite_difference(sequence_kl, [eta])[0],
                        atol=1e-9)
        assert_allclose(full_gradient, frozen_prefix_gradient + prefix_derivative,
                        atol=1e-12)
        assert_allclose(immediate_only, frozen_prefix_gradient, atol=1e-12)
        if eta == 0:
            assert_allclose(full_gradient, branch_kl / 4, atol=1e-12)
            assert_allclose(frozen_prefix_gradient, 0., atol=1e-12)
            print(f"Sequence KL: full gradient = {full_gradient:.10f}; "
                  f"frozen-prefix gradient = {frozen_prefix_gradient:.1f}")


def check_gaussian_estimators():
    z, weights = normal_quadrature()
    for mu, log_scale, target_mean, target_scale in [
        (-.7, -.4, 1.1, .9), (1.2, .3, -.2, 1.7), (0., 0., 0., 1.)
    ]:
        scale = np.exp(log_scale)
        target_var = target_scale ** 2
        x = mu + scale * z
        logq = -.5 * np.log(2 * np.pi) - log_scale - .5 * z ** 2
        logp = (-.5 * np.log(2 * np.pi) - np.log(target_scale)
                - .5 * (x - target_mean) ** 2 / target_var)
        # Both columns are derivatives at fixed x, not through the sampler.
        parameter_score = np.column_stack((z / scale, z ** 2 - 1))
        jacobian = np.column_stack((np.ones_like(z), scale * z))
        score_q = -z / scale
        score_p = -(x - target_mean) / target_var
        likelihood_ratio = np.sum(
            (weights * (logq - logp))[:, None] * parameter_score, axis=0)
        pathwise = np.sum(
            (weights * (score_q - score_p))[:, None] * jacobian, axis=0)
        analytic = np.array([(mu - target_mean) / target_var,
                             scale ** 2 / target_var - 1])

        def gaussian_kl(parameters):
            mean, log_std = parameters
            return (np.log(target_scale) - log_std
                    + (np.exp(2 * log_std) + (mean - target_mean) ** 2)
                    / (2 * target_var) - .5)

        assert_allclose(likelihood_ratio, analytic, atol=1e-11)
        assert_allclose(pathwise, analytic, atol=1e-11)
        assert_allclose(finite_difference(gaussian_kl, [mu, log_scale]), analytic,
                        atol=1e-9)
        entropy_ascent = np.sum(
            (-weights * score_q)[:, None] * jacobian, axis=0)
        assert_allclose(entropy_ascent, [0., 1.], atol=1e-12)
    print("Gaussian mean + log-scale: likelihood ratio = pathwise = finite "
          "difference; entropy ascent = [0, 1]")


def check_noisy_point_mass():
    # A point-mass clean student has no density; its noisy law does.
    epsilon, weights = normal_quadrature()
    theta, target_mean, target_var = -.7, 1.2, .8 ** 2
    for alpha, sigma in [(.95, .1), (.6, .7), (.1, 1.4)]:
        variance_t = alpha ** 2 * target_var + sigma ** 2
        xt = alpha * theta + sigma * epsilon
        score_q = -epsilon / sigma
        score_t = -(xt - alpha * target_mean) / variance_t
        epsilon_q = epsilon  # Here x0 is deterministic, so this equals E[eps|xt].
        epsilon_t = -sigma * score_t
        score_gradient = np.sum(weights * alpha * (score_q - score_t))
        epsilon_gradient = np.sum(
            weights * alpha / sigma * (epsilon_t - epsilon_q))
        analytic = alpha ** 2 * (theta - target_mean) / variance_t

        def noisy_kl(parameters):
            return (.5 * np.log(variance_t / sigma ** 2)
                    + (sigma ** 2 + alpha ** 2 * (parameters[0] - target_mean) ** 2)
                    / (2 * variance_t) - .5)

        assert_allclose([score_gradient, epsilon_gradient], analytic, atol=1e-12)
        assert_allclose(finite_difference(noisy_kl, [theta])[0], analytic,
                        atol=1e-9)
        assert noisy_kl([theta - .05 * epsilon_gradient]) < noisy_kl([theta])
    print("Noisy point mass: teacher-minus-fake epsilon gives the LOSS gradient; "
          "subtracting it decreases KL")


def check_multivariate_gaussian_opd():
    # A nonlinear 2D mean with a nontrivial Jacobian tests the vector PG -> MSE.
    nodes, weights = normal_quadrature()
    z1, z2 = np.meshgrid(nodes, nodes, indexing="ij")
    epsilon = np.column_stack((z1.ravel(), z2.ravel()))
    joint_weights = np.outer(weights, weights).ravel()
    parameters = np.array([.4, -.6])
    target_mean = np.array([-.2, .8])

    def mean(theta):
        return np.array([theta[0] ** 2 + theta[1], np.sin(theta[0]) - theta[1]])

    jacobian = np.array([[2 * parameters[0], 1.],
                         [np.cos(parameters[0]), -1.]])
    delta = mean(parameters) - target_mean
    for sigma in [.25, .8, 1.7]:
        constant = delta @ delta / (2 * sigma ** 2)
        sampled_cost = constant + np.sum(epsilon * delta, axis=1) / sigma
        parameter_score = np.sum((epsilon / sigma)[:, :, None]
                                 * jacobian[None, :, :], axis=1)
        pg = np.sum((joint_weights * sampled_cost)[:, None] * parameter_score,
                    axis=0)
        baseline_only = np.sum((joint_weights * constant)[:, None]
                               * parameter_score, axis=0)
        analytic = jacobian.T @ delta / sigma ** 2

        def mean_mse(theta):
            difference = mean(theta) - target_mean
            return difference @ difference / (2 * sigma ** 2)

        assert_allclose(pg, analytic, atol=1e-10)
        assert_allclose(finite_difference(mean_mse, parameters), analytic,
                        atol=1e-8)
        assert_allclose(baseline_only, [0., 0.], atol=1e-11)
    print("2D Gaussian OPD: sampled log-ratio PG = mean MSE; "
          "detached averaged-KL reward gives zero conditional gradient")


def check_discriminator_gradients():
    z, weights = normal_quadrature()
    theta0 = np.array([-.7, -.3])  # mean and log standard deviation
    target_mean, target_scale = .9, 1.1
    scale0 = np.exp(theta0[1])
    x = theta0[0] + scale0 * z
    jacobian = np.column_stack((np.ones_like(z), scale0 * z))

    def frozen_logit(points):
        # The discriminator is optimal for theta0 and remains fixed during G's step.
        log_target = (-np.log(target_scale)
                      - (points - target_mean) ** 2 / (2 * target_scale ** 2))
        log_student0 = (-theta0[1] - (points - theta0[0]) ** 2 / (2 * scale0 ** 2))
        return log_target - log_student0

    logit = frozen_logit(x)
    probability = 1 / (1 + np.exp(-logit))
    score_difference = (-(x - target_mean) / target_scale ** 2
                        + (x - theta0[0]) / scale0 ** 2)
    input_fd = (frozen_logit(x + 1e-5) - frozen_logit(x - 1e-5)) / 2e-5
    assert_allclose(input_fd, score_difference, atol=1e-8)
    gradients = {}
    for name, multiplier in [("negative_logit", np.ones_like(x)),
                             ("minimax", probability),
                             ("non_saturating", 1 - probability)]:
        def generator_loss(parameters):
            logits = frozen_logit(parameters[0] + np.exp(parameters[1]) * z)
            losses = {"negative_logit": -logits,
                      "minimax": -np.logaddexp(0., logits),
                      "non_saturating": np.logaddexp(0., -logits)}
            return weights @ losses[name]

        analytic = -np.sum((weights * multiplier * score_difference)[:, None]
                           * jacobian, axis=0)
        assert_allclose(finite_difference(generator_loss, theta0), analytic,
                        atol=1e-8)
        gradients[name] = analytic
    exact_kl_gradient = np.array([(theta0[0] - target_mean) / target_scale ** 2,
                                  scale0 ** 2 / target_scale ** 2 - 1])
    assert_allclose(gradients["negative_logit"], exact_kl_gradient, atol=1e-11)
    assert not np.allclose(gradients["non_saturating"], exact_kl_gradient)
    assert not np.allclose(gradients["minimax"], exact_kl_gradient)
    print("GAN: ideal logit input gradient = score difference; "
          "negative-logit, minimax, and non-saturating generator gradients verified")


def check_reward_tilt_identity():
    reference = np.array([.2, .5, .3])
    student = np.array([.4, .1, .5])
    reward = np.array([-1., .8, 1.3])
    for beta in [.3, 1., 2.5]:
        unnormalized = reference * np.exp(reward / beta)
        partition = np.sum(unnormalized)
        target = unnormalized / partition
        objective = (student @ reward
                     - beta * np.sum(student * np.log(student / reference)))
        kl_form = (-beta * np.sum(student * np.log(student / target))
                   + beta * np.log(partition))
        assert_allclose(objective, kl_form, atol=1e-12)
    print("KL-regularized reward: reward-tilted target identity verified")


if __name__ == "__main__":
    check_sequence_prefix_gradient()
    check_gaussian_estimators()
    check_noisy_point_mass()
    check_multivariate_gaussian_opd()
    check_discriminator_gradients()
    check_reward_tilt_identity()
    print("All checks passed.")
