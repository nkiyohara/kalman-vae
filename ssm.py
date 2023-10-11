import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMModel(nn.Module):
    def __init__(self, a_dim, K):
        super(LSTMModel, self).__init__()
        self.a_dim = a_dim
        self.K = K
    
        self.lstm = nn.LSTM(a_dim, K, batch_first=False)

    def forward(self, x):
        x, h = self.lstm(x)
        x = F.softmax(x, dim=-1)
        return x


class StateSpaceModel(nn.Module):
    def __init__(self, a_dim, z_dim, K, Q_reg=1e-3, R_reg=1e-3):
        
        super(StateSpaceModel, self).__init__()
        
        self.a_dim = a_dim
        self.z_dim = z_dim
        self.K = K
        
        self.mat_A_K = nn.Parameter(torch.randn(K, z_dim, z_dim))
        self.mat_C_K = nn.Parameter(torch.randn(K, a_dim, z_dim))
        self.mat_Q_L = nn.Parameter(torch.randn(z_dim, z_dim))
        self.mat_R_L = nn.Parameter(torch.randn(a_dim, a_dim))
        self.Q_reg = Q_reg
        self.R_reg = R_reg
        
        self.weight_model = LSTMModel(a_dim, K)
        # input shape: (sequence_length, batch_size, a_dim)
        # output shape: (sequence_length, batch_size, K)
    
    @property
    def mat_Q(self):
        # shape: (z_dim, z_dim)
        return self.mat_Q_L @ self.mat_Q_L.T + torch.eye(self.z_dim)*self.Q_reg
    
    @property
    def mat_R(self):
        # shape: (a_dim, a_dim)
        return self.mat_R_L @ self.mat_R_L.T + torch.eye(self.a_dim)*self.R_reg
    
    def kalman_filter(self, as_, initial_std=1.0):
        # as_: a_0, a_1, ..., a_{T-1}
        
        sequence_length, batch_size = as_.size()[:2]
        
        # Initial state estimate: \hat{z}_{0|-1}
        mean_t_plus = torch.zeros(batch_size, self.z_dim, 1)
        # Initial state covariance: \Sigma_{0|-1}
        cov_t_plus = (torch.eye(self.z_dim)*initial_std**2).unsqueeze(0).repeat(batch_size, 1, 1)  
        
        weights = self.weight_model(as_)
        # Shape of weights is (sequence_length, batch_size, K)
        # Shape of mat_As and mat_Cs is (sequence_length, batch_size, z_dim, z_dim)
        # A_0, A_1, ..., A_{T-1}
        mat_As = torch.einsum('tbk,kij->tbij', weights, self.mat_A_K)
        # C_0, C_1, ..., C_{T-1}
        mat_Cs = torch.einsum('tbk,kij->tbij', weights, self.mat_C_K)
        
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
            K_t = cov_t_plus @ mat_Cs[t].transpose(1, 2) @ torch.inverse(mat_Cs[t] @ cov_t_plus @ mat_Cs[t].transpose(1, 2) + self.mat_R)

            # \hat{z}_{0|0}, \hat{z}_{1|1}, ..., \hat{z}_{T-1|T-1}
            mean_t = mean_t_plus + K_t @ (as_[t].unsqueeze(2) - mat_Cs[t] @ mean_t_plus)  # Updated state estimate
            # z_{1|0}, z_{2|1}, ..., z_{T|T-1}
            mean_t_plus = mat_As[t] @ mean_t  # Predicted state estimate

            # \Sigma_{0|0}, \Sigma_{1|1}, ..., \Sigma_{T-1|T-1}
            cov_t = cov_t_plus - K_t @ mat_Cs[t] @ cov_t_plus  # Updated state covariance
            # \Sigma_{1|0}, \Sigma_{2|1}, ..., \Sigma_{T|T-1}
            cov_t_plus = mat_As[t] @ cov_t @ mat_As[t].transpose(1, 2) + self.mat_Q  # Predicted state covariance

            means.append(mean_t)
            covariances.append(cov_t)
            next_means.append(mean_t_plus)
            next_covariances.append(cov_t_plus)

        return means, covariances, next_means, next_covariances, mat_As, mat_Cs
    
    def kalman_smooth(self, as_, filter_means, filter_covariances, filter_next_means, filter_next_covariances, mat_As, mat_Cs):

        # import pdb; pdb.set_trace()
        sequence_length, batch_size, _ = as_.size()

        means = [filter_means[-1]]  # \hat{z}_{T-1|T-1}
        covariances = [filter_covariances[-1]]  # \Sigma_{T-1|T-1}

        for t in reversed(range(sequence_length - 1)):

            # J_{T-2}, J_{T-3}, ..., J_0
            J_t = filter_covariances[t] @ mat_As[t+1].transpose(1, 2) @ torch.inverse(filter_next_covariances[t])
            
            # \hat{z}_{T-2}, \hat{z}_{T-3}, ..., \hat{z}_0
            mean_t = filter_means[t] + J_t @ (means[0] - filter_next_means[0])
            # \Sigma_{T-2}, \Sigma_{T-3}, ..., \Sigma_0
            cov_t = filter_covariances[t] + J_t @ (covariances[0] - filter_next_covariances[t]) @ J_t.transpose(1, 2)

            means.insert(0, mean_t)
            covariances.insert(0, cov_t)

        return torch.stack(means), torch.stack(covariances)