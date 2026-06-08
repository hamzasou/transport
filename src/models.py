import numpy as np
import torch
import torch.nn as nn


class SWFlowModel(nn.Module):
    """
    Sliced-Wasserstein Normalizing Flow model.

    Args:
        flows (list): liste des transformations/flows à appliquer.
        device (str or torch.device): device utilisé, par exemple "cpu" ou "cuda".
        noise_std (float): écart-type du bruit gaussien source.

    Si noise_std = 0.7, alors :
        Z ~ N(0, 0.7^2 I)
    """

    def __init__(self, flows, device="cpu", noise_std=1.0):
        super().__init__()

        self.device = torch.device(device)
        self.flows = nn.ModuleList(flows).to(self.device)
        self.nb_flows = len(flows)

        # Écart-type du bruit gaussien source
        self.noise_std = noise_std

    def set_noise_std(self, noise_std):
        """
        Permet de modifier l'écart-type du bruit après création du modèle.
        """

        self.noise_std = noise_std

    def sample_noise(self, nb_samples, dim):
        """
        Génère un bruit gaussien contrôlé :

            Z ~ N(0, noise_std^2 I)

        Exemple :
            noise_std = 0.7 donne Z ~ N(0, 0.7^2 I)
        """

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        z = self.noise_std * torch.randn(
            nb_samples,
            dim,
            device=device,
            dtype=dtype
        )

        return z

    def generate_from_noise(self, nb_samples, dim):
        """
        Génère des données à partir d'un bruit gaussien contrôlé.

        Étapes :
            z ~ N(0, noise_std^2 I)
            y = T(z)
        """

        z = self.sample_noise(nb_samples, dim)

        y, _, _ = self.forward(z)

        return y

    def forward(self, x):
        """
        Passage direct :
            x -> T(x)

        Returns:
            x: sortie transformée
            shatten: régularisation moyenne basée sur log_diag
            log_det: log-déterminant jacobien total
        """

        m, _ = x.shape

        shatten = torch.zeros((), device=x.device, dtype=x.dtype)
        log_det = torch.zeros(m, device=x.device, dtype=x.dtype)

        for flow in self.flows:
            x, log_diag, ld = flow.forward(x)

            # Régularisation plus stable que torch.sum(log_diag ** 2)
            # Ici on prend la moyenne pour éviter une dépendance trop forte
            # au batch_size et à la dimension.
            shatten = shatten + torch.mean(log_diag ** 2)

            log_det = log_det + ld

        return x, shatten, log_det

    def transport_cost(self, x):
        """
        Coût de transport moyen entre chaque étape intermédiaire.

        Pour chaque flow T_i, on calcule :

            E[ ||x - T_i(x)||_2^2 ]

        Ce choix correspond mieux au coût quadratique de Monge que la norme
        de Frobenius globale sur tout le batch.
        """

        total_cost = torch.zeros((), device=x.device, dtype=x.dtype)

        for flow in self.flows:
            z, _, _ = flow.forward(x)

            # Coût quadratique moyen par échantillon
            step_cost = torch.sum((x - z) ** 2, dim=1).mean()

            total_cost = total_cost + step_cost

            x = z

        return total_cost / self.nb_flows

    def forward_barycenter(self, x, nb_flows):
        """
        Applique seulement les nb_flows premiers flows.
        Utile pour visualiser les transports intermédiaires.
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

        Attention :
        cette fonction applique les inverses en partant du dernier flow.
        Elle est utile pour visualiser des transports inverses intermédiaires.
        """

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

        y = torch.from_numpy(data.astype(np.float32)).to(self.device)

        x = self.inverse(y)

        return x

    def sample_y(self, x_sampler, nb_samples):
        """
        Échantillonne x puis applique le modèle direct pour obtenir y.

        x_sampler doit retourner :
            data, label
        """

        data, label = x_sampler(nb_samples)

        x = torch.from_numpy(data.astype(np.float32)).to(self.device)

        y, _, _ = self.forward(x)

        return y