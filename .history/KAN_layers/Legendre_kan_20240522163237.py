import torch
import torch.nn as nn

class RecurrentLegendreLayer(nn.Module):
    def __init__(self, max_degree, input_dim, output_dim):
        super(RecurrentLegendreLayer, self).__init__()
        self.max_degree = max_degree
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Parameters for the linear combination of Legendre polynomials
        # Adjust dimensions: one set of weights for each polynomial degree, across all input/output dimension pairs
        self.weights = nn.Parameter(torch.randn(max_degree + 1, self.input_dim, self.output_dim))
        # nn.init.xavier_normal_(self.weights)
        nn.init.orthogonal_(self.weights)
        # nn.init.kaiming_normal_(self.weights)


        self.dropout = nn.Dropout(.1)
        # Optional: Bias for each output dimension
        self.bias = nn.Parameter(torch.zeros(self.output_dim))

    def forward(self, x):
        batch_size = x.shape[0]

        # Initialize P0 and P1 for the recurrence relation
        P_n_minus_2 = torch.ones((batch_size, self.input_dim), device=x.device)
        P_n_minus_1 = x.clone()

        # Store all polynomial values
        polys = [P_n_minus_2.unsqueeze(-1), P_n_minus_1.unsqueeze(-1)]

        # Compute higher order polynomials up to max_degree
        for n in range(2, self.max_degree + 1):
            P_n = ((2 * n - 1) * x * P_n_minus_1 - (n - 1) * P_n_minus_2) / n
            polys.append(P_n.unsqueeze(-1))
            P_n_minus_2, P_n_minus_1 = P_n_minus_1, P_n

        # Concatenate all polynomial values
        polys = torch.cat(polys, dim=-1)  # Shape: [batch_size, input_dim, max_degree + 1]
        polys = self.dropout(polys)
        # Linearly combine polynomial features
        output = torch.einsum('bif,fio->bo', polys, self.weights) + self.bias

        return output
