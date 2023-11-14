import logging
from typing import Optional

import torch
import torch.distributions as D
import torch.nn as nn

from dynamics_parameter_network import LSTMModel
from misc import _validate_shape
from sample_control import SampleControl

logger = logging.getLogger(__name__)


class StateSpaceModel(nn.Module):
    def __init__(
        self,
        a_dim: int,
        z_dim: int,
        K: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        Q_reg: float = 1e-3,
        R_reg: float = 1e-3,
        init_reg_weight: float = 0.9,
        initial_state_mean: Optional[torch.Tensor] = None,
        initial_state_covariance: Optional[torch.Tensor] = None,
        fix_matrices: bool = False,
    ):
        super(StateSpaceModel, self).__init__()

        self.a_dim = a_dim
        self.z_dim = z_dim
        self.a_0 = nn.Parameter(torch.zeros(a_dim))
        self.K = K

        self._mat_A_K = nn.Parameter(
            (1.0 - init_reg_weight) * torch.randn(K, z_dim, z_dim)
            + init_reg_weight * torch.eye(z_dim)
        )
        self._mat_C_K = nn.Parameter(
            (1.0 - init_reg_weight) * torch.randn(K, a_dim, z_dim)
            + init_reg_weight * torch.eye(a_dim, z_dim)
        )
        mat_Q = (1.0 - init_reg_weight) * torch.randn(
            z_dim, z_dim
        ) + init_reg_weight * torch.eye(z_dim)
        mat_Q = mat_Q @ mat_Q.T
        mat_R = (1.0 - init_reg_weight) * torch.randn(
            a_dim, a_dim
        ) + init_reg_weight * torch.eye(a_dim)
        mat_R = mat_R @ mat_R.T
        self._mat_Q_L = nn.Parameter(torch.linalg.cholesky((mat_Q + mat_Q.T) / 2.0))
        self._mat_R_L = nn.Parameter(torch.linalg.cholesky((mat_R + mat_R.T) / 2.0))
        self._a_eye = torch.eye(a_dim)
        self._z_eye = torch.eye(z_dim)
        self.Q_reg = Q_reg
        self.R_reg = R_reg

        self.fix_matrices = fix_matrices

        if initial_state_mean is None:
            self.initial_state_mean = torch.zeros(z_dim)
        else:
            _validate_shape(initial_state_mean, (z_dim,), "initial_state_mean")
            self.initial_state_mean = initial_state_mean

        if initial_state_covariance is None:
            self.initial_state_covariance = torch.eye(z_dim)
        else:
            _validate_shape(
                initial_state_covariance, (z_dim, z_dim), "initial_state_covariance"
            )
            self.initial_state_covariance = initial_state_covariance

        self.weight_model = LSTMModel(
            a_dim, K, hidden_dim=hidden_dim, num_layers=num_layers
        )
        # input shape: (sequence_length, batch_size, a_dim)
        # output shape: (sequence_length, batch_size, K)

    def _apply(self, fn):
        super()._apply(fn)
        self._a_eye = fn(self._a_eye)
        self._z_eye = fn(self._z_eye)
        self.initial_state_mean = fn(self.initial_state_mean)
        self.initial_state_covariance = fn(self.initial_state_covariance)
        return self

    @property
    def mat_Q(self) -> torch.Tensor:
        # shape: (z_dim, z_dim)
        matrix = self._mat_Q_L @ self._mat_Q_L.T + self._z_eye * self.Q_reg
        if self.fix_matrices:
            matrix = matrix.detach()
        return matrix

    @property
    def mat_R(self) -> torch.Tensor:
        # shape: (a_dim, a_dim)
        matrix = self._mat_R_L @ self._mat_R_L.T + self._a_eye * self.R_reg
        if self.fix_matrices:
            matrix = matrix.detach()
        return matrix

    @property
    def mat_A_K(self) -> torch.Tensor:
        # shape: (K, z_dim, z_dim)
        matrix = self._mat_A_K
        if self.fix_matrices:
            matrix = matrix.detach()
        return matrix

    @property
    def mat_C_K(self) -> torch.Tensor:
        # shape: (K, a_dim, z_dim)
        matrix = self._mat_C_K
        if self.fix_matrices:
            matrix = matrix.detach()
        return matrix

    @mat_Q.setter
    def mat_Q(self, value: torch.Tensor):
        # shape: (z_dim, z_dim)
        _validate_shape(value, (self.z_dim, self.z_dim), "mat_Q")
        self._mat_Q_L = nn.Parameter(torch.linalg.cholesky(value))
        self.Q_reg = 0.0

    @mat_R.setter
    def mat_R(self, value: torch.Tensor):
        # shape: (a_dim, a_dim)
        _validate_shape(value, (self.a_dim, self.a_dim), "mat_R")
        self._mat_R_L = nn.Parameter(torch.linalg.cholesky(value))
        self.R_reg = 0.0

    @mat_A_K.setter
    def mat_A_K(self, value: torch.Tensor):
        _validate_shape(value, (self.K, self.z_dim, self.z_dim), "mat_A_K")
        self._mat_A_K = nn.Parameter(value)

    @mat_C_K.setter
    def mat_C_K(self, value: torch.Tensor):
        _validate_shape(value, (self.K, self.a_dim, self.z_dim), "mat_C_K")
        self._mat_C_K = nn.Parameter(value)

    def kalman_filter(
        self,
        as_: torch.Tensor,
        sample_control: SampleControl,
        observation_mask: Optional[torch.Tensor] = None,
        learn_weight_model: bool = True,
        symmetrize_covariance: bool = True,
        burn_in: int = 0,
    ):
        # as_: a_0, a_1, ..., a_{T-1}
        # shape: (sequence_length, batch_size, a_dim)
        sequence_length, batch_size = as_.size()[:2]

        # Initialize dynamics parameter network
        self.weight_model.clear_hidden_state()

        # Initial weight
        weight_next = (
            torch.ones(1, batch_size, self.K, device=as_.device, dtype=as_.dtype)
            / self.K
        )

        # Initial state estimate: \hat{z}_{0|-1}
        # shape: (batch_size, z_dim)
        mean_t_plus = self.initial_state_mean.repeat(batch_size, 1)

        # Initial state covariance: \Sigma_{0|-1}
        # shape: (batch_size, z_dim, z_dim)
        cov_t_plus = self.initial_state_covariance.unsqueeze(0).repeat(batch_size, 1, 1)

        # \hat{z}_{0|0}, \hat{z}_{1|1}, ..., \hat{z}_{T-1|T-1}
        means = []

        # \Sigma_{0|0}, \Sigma_{1|1}, ..., \Sigma_{T-1|T-1}
        covariances = []

        # z_{1|0}, z_{2|1}, ..., z_{T|T-1}
        next_means = []

        # \Sigma_{1|0}, \Sigma_{2|1}, ..., \Sigma_{T|T-1}
        next_covariances = []

        as_for_weight_list = []

        mat_As_list = []
        mat_Cs_list = []

        for t in range(sequence_length):
            weight = weight_next

            if t == 0:
                mat_A = torch.einsum("tbk,kij->bij", weight, self.mat_A_K)
            else:
                mat_A = mat_A_next
            mat_C = torch.einsum("tbk,kij->bij", weight, self.mat_C_K)

            a_observed = as_[t]
            z_sample = D.MultivariateNormal(
                torch.bmm(mat_A, mean_t_plus.unsqueeze(-1)).squeeze(-1), self.mat_Q
            ).rsample()
            a_unobserved = D.MultivariateNormal(
                torch.bmm(mat_C, z_sample.unsqueeze(-1)).squeeze(-1), self.mat_R
            ).rsample()

            if observation_mask is None:
                a_for_weight = a_observed
            else:
                a_for_weight = (
                    observation_mask[t].unsqueeze(-1) * a_observed
                    + (1.0 - observation_mask[t].unsqueeze(-1)) * a_unobserved
                )
            as_for_weight_list.append(a_for_weight)

            if learn_weight_model:
                weight_next = self.weight_model(a_for_weight.unsqueeze(0))
            else:
                weight_next = self.weight_model(a_for_weight.unsqueeze(0)).detach()

            mat_A_next = torch.einsum("tbk,kij->bij", weight_next, self.mat_A_K)

            mat_As_list.append(mat_A)
            mat_Cs_list.append(mat_C)

            # Kalman gain
            # K_0, K_1, ..., K_{T-1}
            K_t = (
                cov_t_plus
                @ mat_C.transpose(1, 2)
                @ torch.inverse(mat_C @ cov_t_plus @ mat_C.transpose(1, 2) + self.mat_R)
            )

            # \hat{z}_{0|0}, \hat{z}_{1|1}, ..., \hat{z}_{T-1|T-1}
            mean_t_observed = mean_t_plus + torch.bmm(
                K_t,
                (as_[t].unsqueeze(-1) - torch.bmm(mat_C, mean_t_plus.unsqueeze(-1))),
            ).squeeze(
                -1
            )  # Updated state estimate

            mean_t_unobserved = mean_t_plus

            # \Sigma_{0|0}, \Sigma_{1|1}, ..., \Sigma_{T-1|T-1}
            cov_t_observed = (
                cov_t_plus - K_t @ mat_C @ cov_t_plus
            )  # Updated state covariance
            cov_t_unobserved = cov_t_plus

            if observation_mask is None:
                mean_t = mean_t_observed
                cov_t = cov_t_observed
            else:
                mean_t = (
                    observation_mask[t].unsqueeze(-1) * mean_t_observed
                    + (1.0 - observation_mask[t]).unsqueeze(-1) * mean_t_unobserved
                )
                cov_t = (
                    observation_mask[t].unsqueeze(-1).unsqueeze(-1) * cov_t_observed
                    + (1.0 - observation_mask[t]).unsqueeze(-1).unsqueeze(-1)
                    * cov_t_unobserved
                )
            if symmetrize_covariance:
                cov_t = (cov_t + cov_t.transpose(1, 2)) / 2.0

            # z_{1|0}, z_{2|1}, ..., z_{T|T-1}
            mean_t_plus = torch.bmm(mat_A_next, mean_t.unsqueeze(-1)).squeeze(
                -1
            )  # Predicted state estimate

            # \Sigma_{1|0}, \Sigma_{2|1}, ..., \Sigma_{T|T-1}
            cov_t_plus = (
                mat_A_next @ cov_t @ mat_A_next.transpose(1, 2) + self.mat_Q
            )  # Predicted state covariance

            if symmetrize_covariance:
                cov_t_plus = (cov_t_plus + cov_t_plus.transpose(1, 2)) / 2.0

            if t < burn_in:
                mean_t = mean_t.detach()
                cov_t = cov_t.detach()
                mean_t_plus = mean_t_plus.detach()
                cov_t_plus = cov_t_plus.detach()

            means.append(mean_t)
            covariances.append(cov_t)
            next_means.append(mean_t_plus)
            next_covariances.append(cov_t_plus)

        mat_C_next = torch.einsum("tbk,kij->bij", weight_next, self.mat_C_K)

        mat_As_list.append(mat_A_next)
        mat_Cs_list.append(mat_C_next)

        mat_As = torch.stack(mat_As_list, dim=0)
        mat_Cs = torch.stack(mat_Cs_list, dim=0)

        as_for_weight = torch.stack(as_for_weight_list, dim=0)

        return (
            torch.stack(means),
            torch.stack(covariances),
            torch.stack(next_means),
            torch.stack(next_covariances),
            mat_As,
            mat_Cs,
            as_for_weight,
        )

    def kalman_smooth(
        self,
        as_: torch.Tensor,
        filter_means: torch.Tensor,
        filter_covariances: torch.Tensor,
        filter_next_means: torch.Tensor,
        filter_next_covariances: torch.Tensor,
        mat_As: torch.Tensor,
        mat_Cs: torch.Tensor,
        sample_control: SampleControl,
        symmetrize_covariance: bool = True,
        burn_in: int = 0,
    ):
        sequence_length, batch_size, _ = as_.size()

        means = [filter_means[-1]]  # \hat{z}_{T-1|T-1}
        covariances = [filter_covariances[-1]]  # \Sigma_{T-1|T-1}

        z_distrib = D.MultivariateNormal(
            filter_means[-1].view(-1, self.z_dim), filter_covariances[-1]
        )
        if sample_control.state_transition == "sample":
            z = z_distrib.rsample()
        elif sample_control.state_transition == "mean":
            z = z_distrib.mean
        else:
            raise ValueError(
                "Invalid sample_control.state_transition: {}".format(
                    sample_control.state_transition
                )
            )

        a_distrib = D.MultivariateNormal(
            torch.bmm(mat_Cs[-1], z.unsqueeze(-1)).squeeze(-1), self.mat_R
        )
        if sample_control.observation == "sample":
            a = a_distrib.rsample()
        elif sample_control.observation == "mean":
            a = a_distrib.mean
        else:
            raise ValueError(
                "Invalid sample_control.observation: {}".format(
                    sample_control.observation
                )
            )

        zs_list = [z]
        as_list = [a]

        for t in reversed(range(sequence_length - 1)):
            # J_{T-2}, J_{T-3}, ..., J_0
            J_t = (
                filter_covariances[t]
                @ mat_As[t + 1].transpose(1, 2)
                @ torch.inverse(filter_next_covariances[t])
            )

            # \hat{z}_{T-2}, \hat{z}_{T-3}, ..., \hat{z}_0
            mean_t = filter_means[t] + torch.bmm(
                J_t, (means[0] - filter_next_means[t]).unsqueeze(-1)
            ).squeeze(-1)
            # \Sigma_{T-2}, \Sigma_{T-3}, ..., \Sigma_0
            cov_t = filter_covariances[t] + J_t @ (
                covariances[0] - filter_next_covariances[t]
            ) @ J_t.transpose(1, 2)

            if symmetrize_covariance:
                cov_t = (cov_t + cov_t.transpose(1, 2)) / 2.0

            if t < burn_in:
                mean_t = mean_t.detach()
                cov_t = cov_t.detach()

            z_distrib = D.MultivariateNormal(mean_t.view(batch_size, self.z_dim), cov_t)
            if sample_control.state_transition == "sample":
                z = z_distrib.rsample()
            elif sample_control.state_transition == "mean":
                z = z_distrib.mean
            else:
                raise ValueError(
                    "Invalid sample_control.state_transition: {}".format(
                        sample_control.state_transition
                    )
                )

            a_distrib = D.MultivariateNormal(
                torch.bmm(mat_Cs[t], z.unsqueeze(-1)).squeeze(-1), self.mat_R
            )
            if sample_control.observation == "sample":
                a = a_distrib.rsample()
            elif sample_control.observation == "mean":
                a = a_distrib.mean
            else:
                raise ValueError(
                    "Invalid sample_control.observation: {}".format(
                        sample_control.observation
                    )
                )

            zs_list.insert(0, z)
            as_list.insert(0, a)
            means.insert(0, mean_t)
            covariances.insert(0, cov_t)

        return (
            torch.stack(means),
            torch.stack(covariances),
            torch.stack(zs_list),
            torch.stack(as_list),
        )

    def predict_future(
        self,
        as_,
        means,
        covariances,
        next_means,
        next_covariances,
        mat_As,
        mat_Cs,
        num_steps,
        sample_control: SampleControl,
    ):
        # as_: a_0, a_1, ..., a_{T-1}
        # shape: (sequence_length, batch_size, a_dim)

        sequence_length, batch_size = as_.size()[:2]

        as_list = [a for a in as_]  # a_0, a_1, ..., a_{T-1}
        mat_As_list = [mat_A for mat_A in mat_As]  # A_0, A_1, ..., A_T
        mat_Cs_list = [mat_C for mat_C in mat_Cs]  # C_0, C_1, ..., C_T
        means = [
            mean for mean in means
        ]  # \hat{z}_{0|T-1}, \hat{z}_{1|T-1}, ..., \hat{z}_{T-1|T-1}
        covariances = [
            covariance for covariance in covariances
        ]  # \Sigma_{0|T-1}, \Sigma_{1|T-1}, ..., \Sigma_{T-1|T-1}

        # next_means: z_{1|0}, z_{2|1}, ..., z_{T|T-1}
        next_means = [next_mean for next_mean in next_means]
        # next_covariances: \Sigma_{1|0}, \Sigma_{2|1}, ..., \Sigma_{T|T-1}
        next_covariances = [next_covariance for next_covariance in next_covariances]

        as_tensor = torch.stack(as_list, dim=0)

        for t_future in range(num_steps):
            t = sequence_length + t_future  # T, T + 1, ..., T + num_steps - 1

            means.append(next_means[-1])
            covariances.append(next_covariances[-1])

            next_z_distrib = D.MultivariateNormal(
                next_means[-1].view(-1, self.z_dim), next_covariances[-1]
            )

            if sample_control.state_transition == "sample":
                next_z = next_z_distrib.sample()
            elif sample_control.state_transition == "mean":
                next_z = next_z_distrib.mean
            else:
                raise ValueError(
                    "Invalid sample_control.state_transition: {}".format(
                        sample_control.state_transition
                    )
                )

            next_a = mat_Cs_list[t] @ next_z.unsqueeze(-1)
            as_list.append(next_a.view(batch_size, self.a_dim))
            as_tensor = torch.stack(as_list, dim=0)
            weights = self.weight_model(as_tensor)
            mat_A = torch.einsum("bk,kij->bij", weights[-1], self.mat_A_K)
            mat_C = torch.einsum("bk,kij->bij", weights[-1], self.mat_C_K)
            mat_As_list.append(mat_A)
            mat_Cs_list.append(mat_C)
            next_mean = (mat_A @ next_means[-1].unsqueeze(-1)).squeeze(-1)
            if sample_control.state_transition == "sample":
                next_covariance = self.mat_Q.unsqueeze(0).repeat(batch_size, 1, 1)
            elif sample_control.state_transition == "mean":
                # TODO: This should be considered later
                next_covariance = (
                    mat_A @ next_covariances[-1] @ mat_A.transpose(1, 2) + self.mat_Q
                )
            else:
                raise ValueError(
                    "Invalid sample_control.state_transition: {}".format(
                        sample_control.state_transition
                    )
                )
            next_covariance = (next_covariance + next_covariance.transpose(1, 2)) / 2.0
            next_means.append(next_mean)
            next_covariances.append(next_covariance)

        return (
            as_tensor,
            means,
            covariances,
            next_means,
            next_covariances,
            mat_As_list,
            mat_Cs_list,
        )
