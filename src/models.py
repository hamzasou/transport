import numpy as np
import torch
import torch.nn as nn


class SWFlowModel(nn.Module):
    """
    Sliced-Wasserstein Normalizing Flow model.
    """

    def __init__(self, flows, device="cpu"):
        super().__init__()

        self.device = device
        self.flows = nn.ModuleList(flows).to(self.device)
        self.nb_flows = len(flows)

    def forward(self, x, return_cost=False):
        """
        Passage direct :
            x -> T(x)

        Si return_cost=False :
            retourne x, shatten, log_det

        Si return_cost=True :
            retourne x, shatten, log_det, transport_cost
        """

        m, _ = x.shape

        shatten = torch.zeros((), device=x.device)
        log_det = torch.zeros(m, device=x.device)
        transport_cost = torch.zeros((), device=x.device)

        for flow in self.flows:
            x_old = x

            x, log_diag, ld = flow.forward(x)

            # Régularisation normalisée
            shatten = shatten + torch.mean(log_diag ** 2)

            # Coût de transport normalisé
            if return_cost:
                transport_cost = transport_cost + torch.mean((x - x_old) ** 2)

            log_det = log_det + ld

        shatten = shatten / self.nb_flows

        if return_cost:
            transport_cost = transport_cost / self.nb_flows
            return x, shatten, log_det, transport_cost

        return x, shatten, log_det

    def transport_cost(self, x):
        """
        Coût de transport normalisé entre les étapes intermédiaires.
        """

        cost = torch.zeros(self.nb_flows, device=x.device)

        for i, flow in enumerate(self.flows):
            z, _, _ = flow.forward(x)

            cost[i] = torch.mean((x - z) ** 2)

            x = z

        return cost

    def forward_barycenter(self, x, nb_flows):
        """
        Applique seulement les nb_flows premiers flows.
        """

        for flow in self.flows[:nb_flows]:
            x, _, _ = flow.forward(x)

        return x

    def inverse(self, z):
        """
        Passage inverse :
            z -> T^{-1}(z)
        """

        for flow in reversed(self.flows):
            z, _ = flow.inverse(z)

        return z

    def inverse_barycenter(self, x, nb_flows):
        """
        Applique seulement nb_flows inverses.
        """

        reversed_flows = list(reversed(self.flows))

        for flow in reversed_flows[:nb_flows]:
            x, _ = flow.inverse(x)

        return x

    def sample_x(self, y_sampler, nb_samples):
        """
        Échantillonne y puis applique l'inverse pour obtenir x.
        """

        data, label = y_sampler(nb_samples)

        if isinstance(data, torch.Tensor):
            y = data.float().to(self.device)
        else:
            y = torch.from_numpy(data.astype(np.float32)).to(self.device)

        x = self.inverse(y)

        return x

    def sample_y(self, x_sampler, nb_samples):
        """
        Échantillonne x puis applique le modèle direct pour obtenir y.
        """

        data, label = x_sampler(nb_samples)

        if isinstance(data, torch.Tensor):
            x = data.float().to(self.device)
        else:
            x = torch.from_numpy(data.astype(np.float32)).to(self.device)

        y, _, _ = self.forward(x)

        return y


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, x):
        return x + 0.1 * self.net(x)


class MLP(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim=256,
        num_res_blocks=1
    ):
        super().__init__()

        layers = [
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        ]

        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(hidden_dim))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(hidden_dim, out_dim))

        self.net = nn.Sequential(*layers)

        # Initialisation proche de l'identité
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class StableRealNVP(nn.Module):
    """
    Couche RealNVP stable.

    Paramètres conseillés pour MNIST :
        hidden_dim = 256
        num_res_blocks = 1
        scale_clip = 1.0
    """

    def __init__(
        self,
        dim=784,
        hidden_dim=256,
        num_res_blocks=1,
        scale_clip=1.0,
        use_random_permutation=True
    ):
        super().__init__()

        self.dim = dim
        self.lower_dim = dim // 2
        self.upper_dim = dim - self.lower_dim

        self.scale_clip = scale_clip
        self.use_random_permutation = use_random_permutation

        self.t1 = MLP(
            self.lower_dim,
            self.upper_dim,
            hidden_dim=hidden_dim,
            num_res_blocks=num_res_blocks
        )

        self.s1 = MLP(
            self.lower_dim,
            self.upper_dim,
            hidden_dim=hidden_dim,
            num_res_blocks=num_res_blocks
        )

        self.t2 = MLP(
            self.upper_dim,
            self.lower_dim,
            hidden_dim=hidden_dim,
            num_res_blocks=num_res_blocks
        )

        self.s2 = MLP(
            self.upper_dim,
            self.lower_dim,
            hidden_dim=hidden_dim,
            num_res_blocks=num_res_blocks
        )

        if use_random_permutation:
            perm = torch.randperm(dim)
            inv_perm = torch.argsort(perm)

            self.register_buffer("perm", perm)
            self.register_buffer("inv_perm", inv_perm)

    def forward(self, x):
        """
        Passage direct :
            x -> z
        """

        if self.use_random_permutation:
            x_work = x[:, self.perm]
        else:
            x_work = x

        lower = x_work[:, :self.lower_dim]
        upper = x_work[:, self.lower_dim:]

        # Couplage 1 : upper dépend de lower
        s1 = self.scale_clip * torch.tanh(self.s1(lower))
        t1 = self.t1(lower)

        upper = upper * torch.exp(s1) + t1

        # Couplage 2 : lower dépend de upper
        s2 = self.scale_clip * torch.tanh(self.s2(upper))
        t2 = self.t2(upper)

        lower = lower * torch.exp(s2) + t2

        z_work = torch.cat([lower, upper], dim=1)

        if self.use_random_permutation:
            z = z_work[:, self.inv_perm]
        else:
            z = z_work

        log_diag = torch.cat([s2, s1], dim=1)
        log_det = torch.sum(s1, dim=1) + torch.sum(s2, dim=1)

        return z, log_diag, log_det

    def inverse(self, z):
        """
        Passage inverse :
            z -> x
        """

        if self.use_random_permutation:
            z_work = z[:, self.perm]
        else:
            z_work = z

        lower = z_work[:, :self.lower_dim]
        upper = z_work[:, self.lower_dim:]

        # Inverse du deuxième couplage
        s2 = self.scale_clip * torch.tanh(self.s2(upper))
        t2 = self.t2(upper)

        lower = (lower - t2) * torch.exp(-s2)

        # Inverse du premier couplage
        s1 = self.scale_clip * torch.tanh(self.s1(lower))
        t1 = self.t1(lower)

        upper = (upper - t1) * torch.exp(-s1)

        x_work = torch.cat([lower, upper], dim=1)

        if self.use_random_permutation:
            x = x_work[:, self.inv_perm]
        else:
            x = x_work

        log_det = -torch.sum(s1, dim=1) - torch.sum(s2, dim=1)

        return x, log_det