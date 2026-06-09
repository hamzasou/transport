import numpy as np
import torch
import torch.nn as nn


class SWFlowModel(nn.Module):
    """
    Sliced-Wasserstein Normalizing Flow model.

    Args:
        flows (list): liste des transformations/flows à appliquer.
        device (str or torch.device): device utilisé, par exemple "cpu" ou "cuda".
    """

    def __init__(self, flows, device="cpu"):
        super().__init__()

        self.device = torch.device(device)
        self.flows = nn.ModuleList(flows).to(self.device)
        self.nb_flows = len(flows)

        if self.nb_flows == 0:
            raise ValueError("La liste des flows ne doit pas être vide.")

    def forward(self, x, return_cost=False):
        """
        Passage direct :
            x -> T(x)

        Returns:
            x: sortie transformée
            shatten: régularisation moyenne basée sur log_diag
            log_det: log-déterminant jacobien total

        Si return_cost=True :
            retourne aussi le coût de transport moyen.
        """

        if x.dim() != 2:
            raise ValueError("x doit être de taille (N, d).")

        m, _ = x.shape

        shatten_values = []
        cost_values = []

        log_det = torch.zeros(
            m,
            device=x.device,
            dtype=x.dtype
        )
        for flow in self.flows:
            x_old = x

            x, log_diag, ld = flow.forward(x)

            # Régularisation normalisée
            # Avant : torch.sum(log_diag ** 2)
            # Maintenant : moyenne pour éviter la dépendance au batch size
            shatten_values.append(torch.mean(torch.pow(log_diag, 2)))

            # Coût de transport moyen par échantillon
            # Utilisé seulement si return_cost=True
            cost_i = torch.mean(torch.sum(torch.pow(x - x_old, 2), dim=1))
            cost_values.append(cost_i)

            log_det = log_det + ld

        shatten = torch.stack(shatten_values).mean()

        if return_cost:
            cost = torch.stack(cost_values).mean()
            return x, shatten, log_det, cost

        return x, shatten, log_det

    def transport_cost(self, x):
        """
        Coût de transport entre chaque étape intermédiaire.

        Pour chaque flow T_i, on calcule une moyenne par batch :
            E[ ||T_i(x) - x||_2^2 ]

        Returns:
            cost: tensor de taille (nb_flows,)
        """

        cost = torch.zeros(
            self.nb_flows,
            device=x.device,
            dtype=x.dtype
        )

        for i, flow in enumerate(self.flows):
            x_old = x

            z, _, _ = flow.forward(x)

            # Correction importante :
            # Avant : torch.norm(x - z, p="fro")
            # Maintenant : coût moyen par échantillon
            cost[i] = torch.mean(torch.sum(torch.pow(z - x_old, 2), dim=1))

            x = z

        return cost

    def forward_barycenter(self, x, nb_flows):
        """
        Applique seulement les nb_flows premiers flows.
        Utile pour visualiser les transports intermédiaires.
        """

        if nb_flows < 0 or nb_flows > self.nb_flows:
            raise ValueError("nb_flows doit être entre 0 et self.nb_flows.")

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

        if nb_flows < 0 or nb_flows > self.nb_flows:
            raise ValueError("nb_flows doit être entre 0 et self.nb_flows.")

        reversed_flows = list(reversed(self.flows))

        for flow in reversed_flows[:nb_flows]:
            x, _ = flow.inverse(x)

        return x

    def sample_x(self, y_sampler, nb_samples):
        """
        Échantillonne y puis applique l'inverse pour obtenir x.

        y_sampler doit retourner :
            data, label
        """

        data, label = y_sampler(nb_samples)

        if torch.is_tensor(data):
            y = data.to(
                device=self.device,
                dtype=torch.float32
            )
        else:
            y = torch.from_numpy(
                data.astype(np.float32)
            ).to(self.device)

        x = self.inverse(y)

        return x

    def sample_y(self, x_sampler, nb_samples):
        """
        Échantillonne x puis applique le modèle direct pour obtenir y.

        x_sampler doit retourner :
            data, label
        """

        data, label = x_sampler(nb_samples)

        if torch.is_tensor(data):
            x = data.to(
                device=self.device,
                dtype=torch.float32
            )
        else:
            x = torch.from_numpy(
                data.astype(np.float32)
            ).to(self.device)

        y, _, _ = self.forward(x)

        return y