import matplotlib.pyplot as plt
import torch
from abc import ABC, abstractmethod


def bmv(m, v):
    return (m @ v.unsqueeze(2)).squeeze(2)


def b_dot(v_1, v_2):
    return (v_1 * v_2).sum(dim=1)


class DistHMC(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def grad_log(self, sample):
        pass

    @abstractmethod
    def log_cancel_const(self, sample):
        pass


class NormHMC(DistHMC):
    def __init__(self, loc, covariance_matrix):
        self.p = torch.distributions.MultivariateNormal(loc, covariance_matrix)

    def grad_log(self, sample):
        precision_matrix = self.p.precision_matrix
        return -bmv(precision_matrix, sample - self.p.mean)

    def log_cancel_const(self, sample):
        precision_matrix = self.p.precision_matrix
        c_sample = sample - self.p.mean
        return -0.5 * b_dot(c_sample, bmv(precision_matrix, c_sample))


def leapfrog(p, sample, momentum, mass_matrix_inv, n_steps, step_size):
    for _ in range(n_steps):
        momentum_half_step = momentum - step_size / 2 * p.grad_log(sample)
        sample = sample + step_size * bmv(mass_matrix_inv, momentum_half_step)
        momentum = momentum_half_step - step_size / 2 * p.grad_log(sample)

    return sample, momentum


def hmc(p, sample_0, mass_matrix, n_samples, n_steps, step_size):
    batch_size, dim = sample_0.shape
    samples = torch.zeros((batch_size, n_samples + 1, dim))
    weights = torch.ones((batch_size, n_samples + 1))
    acceptance_p = torch.zeros((batch_size,))
    samples[:, 0] = sample_0
    mass_matrix_inv = torch.inverse(mass_matrix)
    p_momentum = torch.distributions.MultivariateNormal(
        torch.zeros((batch_size, dim)), mass_matrix
    )
    for n in range(1, n_samples + 1):
        sample = samples[:, n - 1]
        momentum_init = p_momentum.rsample(torch.Size([1])).squeeze(0)
        new_sample, momentum = leapfrog(
            p, sample, momentum_init, mass_matrix_inv, n_steps, step_size
        )
        kinetic_diff = 0.5 * (
                b_dot(momentum_init, bmv(mass_matrix_inv, momentum_init))
                - b_dot(momentum, bmv(mass_matrix_inv, momentum))
        )
        potential_diff = \
            p.log_cancel_const(new_sample) - p.log_cancel_const(sample)
        alpha = torch.exp(kinetic_diff + potential_diff)
        weights[:, n] = alpha
        accepted = torch.rand(1) < alpha
        rejected = torch.logical_not(accepted)
        acceptance_p += accepted
        samples[accepted, n] = new_sample[accepted]
        samples[rejected, n] = samples[rejected, n - 1]

    return samples, weights, acceptance_p / n_samples
