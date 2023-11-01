import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

from ssm import StateSpaceModel
from vae import BernoulliDecoder, Encoder, GaussianDecoder


class KalmanVariationalAutoencoder(nn.Module):
    def __init__(
        self, image_size, image_channels, a_dim, z_dim, K, decoder_type="gaussian"
    ):
        super(KalmanVariationalAutoencoder, self).__init__()
        self.encoder = Encoder(image_size, image_channels, a_dim)
        if decoder_type == "gaussian":
            self.decoder = GaussianDecoder(a_dim, image_size, image_channels)
        elif decoder_type == "bernoulli":
            self.decoder = BernoulliDecoder(a_dim, image_size, image_channels)
        else:
            raise ValueError("Unknown decoder type: {}".format(decoder_type))
        self.state_space_model = StateSpaceModel(a_dim=a_dim, z_dim=z_dim, K=K)
        self.a_dim = a_dim
        self.z_dim = z_dim
        self.register_buffer('_zero_val', torch.tensor(0.0)) 

    def elbo(self, xs, reconst_weight=0.3, regularization_weight=1.0, kalman_weight=1.0, kl_weight=0.0, learn_weight_model=True):
        seq_length = xs.shape[0]
        batch_size = xs.shape[1]

        as_dist = self.encoder(xs.reshape(-1, *xs.shape[2:]))
        as_sample = as_dist.rsample().view(seq_length, batch_size, self.a_dim)

        # Reconstruction objective
        xs_dist = self.decoder(as_sample.view(-1, self.a_dim))
        reconstruction_obj = (
            xs_dist.log_prob(xs.reshape(-1, *xs.shape[2:])).sum(0).mean(0).sum()
        )

        # Regularization objective
        # -ln q_\phi(a|x)
        regularization_obj = (
            -as_dist.log_prob(as_sample.view(-1, self.a_dim)).sum(0).mean(0).sum()
        )

        # Kalman filter and smoother
        (
            filter_means,
            filter_covariances,
            filter_next_means,
            filter_next_covariances,
            mat_As,
            mat_Cs,
        ) = self.state_space_model.kalman_filter(as_sample, learn_weight_model)
        means, covariances = self.state_space_model.kalman_smooth(
            as_sample,
            filter_means,
            filter_covariances,
            filter_next_means,
            filter_next_covariances,
            mat_As,
            mat_Cs,
        )

        # Sample from p_\gamma (z|a,u)
        # Shape of means: (sequence_length, batch_size, z_dim, 1)
        # Shape of covariances: (sequence_length, batch_size, z_dim, z_dim)
        zs_distrib = D.MultivariateNormal(
            means.view(-1, self.z_dim), covariances.view(-1, self.z_dim, self.z_dim)
        )

        # KL divergence between q_\phi(a|x) and p(z) for VAE validation purposes
        if kl_weight != 0.0:
            prior_distrib = D.Normal(
                torch.zeros(self.a_dim), torch.ones(self.a_dim)
            )
            kl_reg = -torch.distributions.kl.kl_divergence(
                as_dist, prior_distrib
            ).view(seq_length, batch_size, self.a_dim).sum(0).mean(0).sum(0)
        else:
            kl_reg = self._zero_val


        # For testing purposes
        # zs_distrib = D.MultivariateNormal(torch.stack(filter_means).view(-1, self.z_dim), torch.stack(filter_covariances).view(-1, self.z_dim, self.z_dim))

        zs_sample = zs_distrib.rsample()
        zs_sample = zs_sample.view(seq_length, batch_size, self.z_dim, 1)

        # ln p_\gamma(a|z)
        kalman_observation_distrib = D.MultivariateNormal(
            (mat_Cs @ zs_sample).view(-1, self.a_dim), self.state_space_model.mat_R
        )
        kalman_observation_log_likelihood = (
            kalman_observation_distrib.log_prob(as_sample.view(-1, self.a_dim))
            .view(seq_length, batch_size, -1)
            .sum(0)
            .mean(0)
            .sum()
        )

        # ln p_\gamma(z)
        kalman_state_transition_log_likelihood = (
            self.state_space_model.state_transition_log_likelihood(zs_sample, mat_As)
        )

        # ln p_\gamma(z|a)
        kalman_posterior_log_likelihood = (
            zs_distrib.log_prob(zs_sample.view(-1, self.z_dim))
            .view(seq_length, batch_size, -1)
            .sum(0)
            .mean(0)
            .sum()
        )

        objective = (
            reconst_weight * reconstruction_obj
            + regularization_weight * regularization_obj
            + kl_weight * kl_reg
            + kalman_weight * (
                kalman_observation_log_likelihood
                + kalman_state_transition_log_likelihood
                - kalman_posterior_log_likelihood
            )
        )

        return objective, {
            "reconst_weight": reconst_weight,
            "regularization_weight": regularization_weight,
            "kalman_weight": kalman_weight,
            "kl_weight": kl_weight,
            "reconstruction": reconstruction_obj.cpu().detach().numpy(),
            "regularization": regularization_obj.cpu().detach().numpy(),
            "kl": kl_reg.cpu().detach().numpy(),
            "kalman_observation_log_likelihood": kalman_observation_log_likelihood.cpu().detach().numpy(),
            "kalman_state_transition_log_likelihood": kalman_state_transition_log_likelihood.cpu().detach().numpy(),
            "kalman_posterior_log_likelihood": kalman_posterior_log_likelihood.cpu().detach().numpy(),
        }
