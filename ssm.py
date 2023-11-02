import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F


class LSTMModel(nn.Module):
    def __init__(self, a_dim, K, hidden_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.a_dim = a_dim
        self.K = K

        self.lstm = nn.LSTM(a_dim, hidden_dim, num_layers=num_layers, batch_first=False)
        self.linear = nn.Linear(hidden_dim, K)

    def forward(self, x):
        x, h = self.lstm(x)
        x = self.linear(x)
        x = F.softmax(x, dim=-1)
        return x


class StateSpaceModel(nn.Module):
    def __init__(
        self,
        a_dim,
        z_dim,
        K,
        hidden_dim=128,
        num_layers=2,
        Q_reg=1e-3,
        R_reg=1e-3,
        initial_state_mean=None,
        initial_state_covariance=None,
        fix_matrices=False,
    ):
        super(StateSpaceModel, self).__init__()

        self.a_dim = a_dim
        self.z_dim = z_dim
        self.K = K

        self._mat_A_K = nn.Parameter(torch.randn(K, z_dim, z_dim))
        self._mat_C_K = nn.Parameter(torch.randn(K, a_dim, z_dim))
        self._mat_Q_L = nn.Parameter(torch.randn(z_dim, z_dim))
        self._mat_R_L = nn.Parameter(torch.randn(a_dim, a_dim))
        self._a_eye = torch.eye(a_dim)
        self._z_eye = torch.eye(z_dim)
        self.Q_reg = Q_reg
        self.R_reg = R_reg

        self.fix_matrices = fix_matrices

        if initial_state_mean is None:
            self.initial_state_mean = torch.zeros(z_dim)
        else:
            if initial_state_mean.shape != (z_dim,):
                raise ValueError(
                    "initial_state_mean must have shape (z_dim,), got {}".format(
                        initial_state_mean.shape
                    )
                )
            self.initial_state_mean = initial_state_mean

        if initial_state_covariance is None:
            self.initial_state_covariance = torch.eye(z_dim)
        else:
            if initial_state_covariance.shape != (z_dim, z_dim):
                raise ValueError(
                    "initial_state_covariance must have shape (z_dim, z_dim), got {}".format(
                        initial_state_covariance.shape
                    )
                )
            self.initial_state_covariance = initial_state_covariance

        self.weight_model = LSTMModel(a_dim, K, hidden_dim=hidden_dim, num_layers=num_layers)
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
    def mat_Q(self):
        # shape: (z_dim, z_dim)
        matrix = self._mat_Q_L @ self._mat_Q_L.T + self._z_eye * self.Q_reg
        if self.fix_matrices:
            matrix = matrix.detach()
        return matrix

    @property
    def mat_R(self):
        # shape: (a_dim, a_dim)
        matrix = self._mat_R_L @ self._mat_R_L.T + self._a_eye * self.R_reg
        if self.fix_matrices:
            matrix = matrix.detach()
        return matrix
    
    @property
    def mat_A_K(self):
        # shape: (K, z_dim, z_dim)
        matrix = self._mat_A_K
        if self.fix_matrices:
            matrix = matrix.detach()
        return matrix
    
    @property
    def mat_C_K(self):
        # shape: (K, a_dim, z_dim)
        matrix = self._mat_C_K
        if self.fix_matrices:
            matrix = matrix.detach()
        return matrix
    
    @mat_Q.setter
    def mat_Q(self, value):
        # shape: (z_dim, z_dim)
        if value.shape != (self.z_dim, self.z_dim):
            raise ValueError(
                "mat_Q must have shape (z_dim, z_dim), got {}".format(value.shape)
            )
        self._mat_Q_L = nn.Parameter(torch.linalg.cholesky(value))
        self.Q_reg = 0.0

    @mat_R.setter
    def mat_R(self, value):
        # shape: (a_dim, a_dim)
        if value.shape != (self.a_dim, self.a_dim):
            raise ValueError(
                "mat_R must have shape (a_dim, a_dim), got {}".format(value.shape)
            )
        self._mat_R_L = nn.Parameter(torch.linalg.cholesky(value))
        self.R_reg = 0.0

    @mat_A_K.setter
    def mat_A_K(self, value):
        if value.shape != (self.K, self.z_dim, self.z_dim):
            raise ValueError(
                "mat_A_K must have shape (K, z_dim, z_dim), got {}".format(
                    value.shape
                )
            )
        self._mat_A_K = nn.Parameter(value)
    
    @mat_C_K.setter
    def mat_C_K(self, value):
        if value.shape != (self.K, self.a_dim, self.z_dim):
            raise ValueError(
                "mat_C_K must have shape (K, a_dim, z_dim), got {}".format(
                    value.shape
                )
            )
        self._mat_C_K = nn.Parameter(value)

    def kalman_filter(self, as_, learn_weight_model=True, symmetrize_covariance=True):
        # as_: a_0, a_1, ..., a_{T-1}
        # shape: (sequence_length, batch_size, a_dim)

        sequence_length, batch_size = as_.size()[:2]

        # Initial state estimate: \hat{z}_{0|-1}
        # shape: (batch_size, z_dim, 1)
        mean_t_plus = (
            self.initial_state_mean.unsqueeze(0).repeat(batch_size, 1).unsqueeze(2)
        )

        # Initial state covariance: \Sigma_{0|-1}
        # shape: (batch_size, z_dim, z_dim)
        cov_t_plus = self.initial_state_covariance.unsqueeze(0).repeat(batch_size, 1, 1)

        if learn_weight_model:
            weights = self.weight_model(as_)
        else:
            weights = self.weight_model(as_).detach()

        # Shape of weights is (sequence_length, batch_size, K)
        # Shape of mat_As and mat_Cs is (sequence_length, batch_size, z_dim, z_dim)
        # A_0, A_1, ..., A_{T-1}
        mat_As = torch.einsum("tbk,kij->tbij", weights, self.mat_A_K)
        # C_0, C_1, ..., C_{T-1}
        mat_Cs = torch.einsum("tbk,kij->tbij", weights, self.mat_C_K)

        # \hat{z}_{0|0}, \hat{z}_{1|1}, ..., \hat{z}_{T-1|T-1}
        means = []

        # \Sigma_{0|0}, \Sigma_{1|1}, ..., \Sigma_{T-1|T-1}
        covariances = []

        # z_{1|0}, z_{2|1}, ..., z_{T|T-1}
        next_means = []

        # \Sigma_{1|0}, \Sigma_{2|1}, ..., \Sigma_{T|T-1}
        next_covariances = []

        for t in range(sequence_length):
            # Kalman gain
            # K_0, K_1, ..., K_{T-1}
            K_t = (
                cov_t_plus
                @ mat_Cs[t].transpose(1, 2)
                @ torch.inverse(
                    mat_Cs[t] @ cov_t_plus @ mat_Cs[t].transpose(1, 2) + self.mat_R
                )
            )

            # \hat{z}_{0|0}, \hat{z}_{1|1}, ..., \hat{z}_{T-1|T-1}
            mean_t = mean_t_plus + K_t @ (
                as_[t].unsqueeze(2) - mat_Cs[t] @ mean_t_plus
            )  # Updated state estimate
            # z_{1|0}, z_{2|1}, ..., z_{T|T-1}
            mean_t_plus = mat_As[t] @ mean_t  # Predicted state estimate

            # \Sigma_{0|0}, \Sigma_{1|1}, ..., \Sigma_{T-1|T-1}
            cov_t = (
                cov_t_plus - K_t @ mat_Cs[t] @ cov_t_plus
            )  # Updated state covariance

            if symmetrize_covariance:
                cov_t = (cov_t + cov_t.transpose(1, 2)) / 2.0

            # \Sigma_{1|0}, \Sigma_{2|1}, ..., \Sigma_{T|T-1}
            cov_t_plus = (
                mat_As[t] @ cov_t @ mat_As[t].transpose(1, 2) + self.mat_Q
            )  # Predicted state covariance

            if symmetrize_covariance:
                cov_t_plus = (cov_t_plus + cov_t_plus.transpose(1, 2)) / 2.0

            means.append(mean_t)
            covariances.append(cov_t)
            next_means.append(mean_t_plus)
            next_covariances.append(cov_t_plus)

        return means, covariances, next_means, next_covariances, mat_As, mat_Cs

    def kalman_smooth(
        self,
        as_,
        filter_means,
        filter_covariances,
        filter_next_means,
        filter_next_covariances,
        mat_As,
        mat_Cs,
        symmetrize_covariance=True,
    ):
        # import pdb; pdb.set_trace()
        sequence_length, batch_size, _ = as_.size()

        means = [filter_means[-1]]  # \hat{z}_{T-1|T-1}
        covariances = [filter_covariances[-1]]  # \Sigma_{T-1|T-1}

        for t in reversed(range(sequence_length - 1)):
            # J_{T-2}, J_{T-3}, ..., J_0
            J_t = (
                filter_covariances[t]
                @ mat_As[t].transpose(1, 2)
                @ torch.inverse(filter_next_covariances[t])
            )

            # \hat{z}_{T-2}, \hat{z}_{T-3}, ..., \hat{z}_0
            mean_t = filter_means[t] + J_t @ (means[0] - filter_next_means[t])
            # \Sigma_{T-2}, \Sigma_{T-3}, ..., \Sigma_0
            cov_t = filter_covariances[t] + J_t @ (
                covariances[0] - filter_next_covariances[t]
            ) @ J_t.transpose(1, 2)

            if symmetrize_covariance:
                cov_t = (cov_t + cov_t.transpose(1, 2)) / 2.0

            means.insert(0, mean_t)
            covariances.insert(0, cov_t)

        return torch.stack(means), torch.stack(covariances)

    def state_transition_log_likelihood(self, zs, mat_As):
        sequence_length, batch_size, _, _ = zs.size()

        # Initial state estimate: \hat{z}_{0|-1}
        mean_t_plus = (
            self.initial_state_mean.unsqueeze(0).repeat(batch_size, 1).unsqueeze(2)
        )

        # Initial state covariance: \Sigma_{0|-1}
        cov_t_plus = self.initial_state_covariance.unsqueeze(0).repeat(batch_size, 1, 1)

        state_transition_log_likelihood = 0.0

        for t in range(sequence_length):
            distrib = D.MultivariateNormal(mean_t_plus.view(-1, self.z_dim), cov_t_plus)
            state_transition_log_likelihood += distrib.log_prob(
                zs[t].view(-1, self.z_dim)
            ).sum()

            mean_t_plus = mat_As[t] @ zs[t]
            cov_t_plus = self.mat_Q

            # else:
            #     mean_t_plus = mat_As[t] @ mean_t_plus
            #     cov_t_plus = (
            #         mat_As[t] @ cov_t_plus @ mat_As[t].transpose(1, 2) + self.mat_Q
            #     )
            #     cov_t_plus = (cov_t_plus + cov_t_plus.transpose(1, 2)) / 2.0

        return state_transition_log_likelihood
