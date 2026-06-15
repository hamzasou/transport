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

    Convention utilisée dans ce code :
        source z  ->  T(z) = generated

    Donc pour la génération :
        z ~ N(0, noise_std^2 I)
        generated = T(z)
    """

    def __init__(self, flows, device="cpu", noise_std=0.5):
        super().__init__()

        if flows is None or len(flows) == 0:
            raise ValueError("La liste flows ne doit pas être vide.")

        if noise_std <= 0:
            raise ValueError("noise_std doit être strictement positif.")

        self.device = torch.device(device)
        self.flows = nn.ModuleList(flows).to(self.device)
        self.nb_flows = len(flows)
        self.noise_std = float(noise_std)

    # ========================================================
    # Gestion du bruit source
    # ========================================================

    def set_noise_std(self, noise_std):
        """
        Permet de modifier l'écart-type du bruit après création du modèle.
        """

        if noise_std <= 0:
            raise ValueError("noise_std doit être strictement positif.")

        self.noise_std = float(noise_std)

    # ========================================================
    # Outils internes
    # ========================================================

    def _get_device_dtype(self):
        """
        Récupère le device et le dtype du modèle à partir de ses paramètres.
        """

        try:
            param = next(self.parameters())
            return param.device, param.dtype

        except StopIteration:
            # Cas très rare : modèle sans paramètres.
            return self.device, torch.float32

    def _to_tensor(self, data):
        """
        Convertit data en tenseur PyTorch compatible avec le modèle.

        Accepte :
            - numpy.ndarray
            - torch.Tensor
        """

        device, dtype = self._get_device_dtype()

        if isinstance(data, torch.Tensor):
            return data.to(device=device, dtype=dtype)

        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(device=device, dtype=dtype)

        raise TypeError("data doit être un numpy.ndarray ou un torch.Tensor.")

    def _normalize_log_det(self, ld, batch_size, device, dtype):
        """
        Normalise le log-déterminant retourné par un flow.

        On accepte :
            - scalaire
            - tenseur de taille (N,)
            - tenseur de taille (N, 1)

        On retourne toujours un tenseur de taille (N,).
        """

        if not torch.is_tensor(ld):
            raise TypeError("Le log_det retourné par chaque flow doit être un torch.Tensor.")

        ld = ld.to(device=device, dtype=dtype)

        if ld.ndim == 0:
            return ld.expand(batch_size)

        if ld.numel() == 1:
            return ld.reshape(1).expand(batch_size)

        if ld.numel() == batch_size:
            return ld.reshape(batch_size)

        raise ValueError(
            "Forme invalide pour log_det. "
            f"Attendu scalaire, (N,) ou (N,1), reçu {tuple(ld.shape)}."
        )

    def _check_input_2d(self, x, name="x"):
        """
        Vérifie que l'entrée est de taille (N, d).
        """

        if not torch.is_tensor(x):
            raise TypeError(f"{name} doit être un torch.Tensor.")

        if x.dim() != 2:
            raise ValueError(f"L'entrée {name} doit être de taille (N, d).")

    # ========================================================
    # Échantillonnage du bruit source
    # ========================================================

    def sample_noise(self, nb_samples, dim):
        """
        Génère un bruit gaussien contrôlé :

            Z ~ N(0, noise_std^2 I)
        """

        if nb_samples <= 0:
            raise ValueError("nb_samples doit être strictement positif.")

        if dim <= 0:
            raise ValueError("dim doit être strictement positif.")

        device, dtype = self._get_device_dtype()

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

        Cette fonction est utilisée pour la génération, donc elle désactive
        le calcul des gradients.
        """

        was_training = self.training
        self.eval()

        with torch.no_grad():
            z = self.sample_noise(nb_samples, dim)
            y, _, _ = self.forward(z)

        if was_training:
            self.train()

        return y

    # ========================================================
    # Passage direct
    # ========================================================

    def forward(self, x):
        """
        Passage direct :
            x -> T(x)

        Returns:
            x: sortie transformée
            shatten: régularisation moyenne basée sur log_diag
            log_det: log-déterminant jacobien total, de taille (N,)
        """

        self._check_input_2d(x, name="x")

        device, dtype = self._get_device_dtype()
        x = x.to(device=device, dtype=dtype)

        batch_size = x.size(0)

        shatten = torch.zeros((), device=device, dtype=dtype)
        log_det = torch.zeros(batch_size, device=device, dtype=dtype)

        for i, flow in enumerate(self.flows):
            output = flow.forward(x)

            if not isinstance(output, (tuple, list)) or len(output) != 3:
                raise ValueError(
                    f"Le flow numéro {i} doit retourner exactement : "
                    "(x, log_diag, log_det)."
                )

            x_new, log_diag, ld = output

            if not torch.is_tensor(x_new):
                raise TypeError(f"La sortie x du flow numéro {i} doit être un tenseur.")

            if x_new.shape != x.shape:
                raise ValueError(
                    f"Le flow numéro {i} change la forme des données. "
                    f"Avant : {tuple(x.shape)}, après : {tuple(x_new.shape)}."
                )

            if log_diag is None:
                raise ValueError(f"Le flow numéro {i} doit retourner log_diag.")

            if not torch.is_tensor(log_diag):
                raise TypeError(f"log_diag du flow numéro {i} doit être un tenseur.")

            log_diag = log_diag.to(device=device, dtype=dtype)
            ld = self._normalize_log_det(ld, batch_size, device, dtype)

            if not torch.isfinite(x_new).all():
                raise ValueError(f"NaN ou Inf détecté dans la sortie du flow numéro {i}.")

            if not torch.isfinite(log_diag).all():
                raise ValueError(f"NaN ou Inf détecté dans log_diag du flow numéro {i}.")

            if not torch.isfinite(ld).all():
                raise ValueError(f"NaN ou Inf détecté dans log_det du flow numéro {i}.")

            shatten = shatten + torch.mean(log_diag ** 2)
            log_det = log_det + ld
            x = x_new

        shatten = shatten / self.nb_flows

        return x, shatten, log_det

    # ========================================================
    # Coût de transport
    # ========================================================

    def transport_cost(self, x):
        """
        Coût de transport moyen entre chaque étape intermédiaire.

        Pour chaque flow T_i, on calcule :
            E[ ||x - T_i(x)||_2^2 ]

        Puis on moyenne sur le nombre de flows.

        Remarque :
            Cette fonction refait un passage dans les flows.
            Elle est correcte, mais elle augmente le temps d'entraînement.
        """

        self._check_input_2d(x, name="x")

        device, dtype = self._get_device_dtype()
        x = x.to(device=device, dtype=dtype)

        total_cost = torch.zeros((), device=device, dtype=dtype)

        for i, flow in enumerate(self.flows):
            output = flow.forward(x)

            if not isinstance(output, (tuple, list)) or len(output) != 3:
                raise ValueError(
                    f"Le flow numéro {i} doit retourner exactement : "
                    "(x, log_diag, log_det)."
                )

            z, _, _ = output

            if z.shape != x.shape:
                raise ValueError(
                    f"Le flow numéro {i} change la forme des données dans transport_cost. "
                    f"Avant : {tuple(x.shape)}, après : {tuple(z.shape)}."
                )

            if not torch.isfinite(z).all():
                raise ValueError(
                    f"NaN ou Inf détecté dans transport_cost au flow numéro {i}."
                )

            step_cost = torch.sum((x - z) ** 2, dim=1).mean()
            total_cost = total_cost + step_cost

            x = z

        return total_cost / self.nb_flows

    # ========================================================
    # Transports intermédiaires
    # ========================================================

    def forward_barycenter(self, x, nb_flows):
        """
        Applique seulement les nb_flows premiers flows.

        Utile pour visualiser les transports intermédiaires :
            x_0 -> x_1 -> ... -> x_k

        où k = nb_flows.
        """

        self._check_input_2d(x, name="x")

        if nb_flows < 0 or nb_flows > self.nb_flows:
            raise ValueError(
                f"nb_flows doit être compris entre 0 et {self.nb_flows}."
            )

        device, dtype = self._get_device_dtype()
        x = x.to(device=device, dtype=dtype)

        for i, flow in enumerate(self.flows[:nb_flows]):
            output = flow.forward(x)

            if not isinstance(output, (tuple, list)) or len(output) != 3:
                raise ValueError(
                    f"Le flow numéro {i} doit retourner exactement : "
                    "(x, log_diag, log_det)."
                )

            x, _, _ = output

        return x

    # ========================================================
    # Passage inverse
    # ========================================================

    def inverse(self, z):
        """
        Passage inverse :
            z -> T^{-1}(z)

        Cette fonction applique les inverses dans l'ordre inverse :
            T_M^{-1}, ..., T_2^{-1}, T_1^{-1}
        """

        self._check_input_2d(z, name="z")

        device, dtype = self._get_device_dtype()
        z = z.to(device=device, dtype=dtype)

        for i, flow in enumerate(reversed(self.flows)):
            output = flow.inverse(z)

            if not isinstance(output, (tuple, list)) or len(output) != 2:
                raise ValueError(
                    "Chaque flow.inverse doit retourner exactement : "
                    "(z_inverse, log_det_inverse)."
                )

            z_new, _ = output

            if z_new.shape != z.shape:
                raise ValueError(
                    f"Un flow inverse change la forme des données. "
                    f"Avant : {tuple(z.shape)}, après : {tuple(z_new.shape)}."
                )

            if not torch.isfinite(z_new).all():
                raise ValueError(
                    f"NaN ou Inf détecté dans inverse au flow inversé numéro {i}."
                )

            z = z_new

        return z

    def inverse_barycenter(self, x, nb_flows):
        """
        Applique seulement nb_flows inverses en partant du dernier flow.

        Attention :
            cette fonction applique les inverses à partir de la fin :
                T_M^{-1}, T_{M-1}^{-1}, ...

        Elle est utile si x est dans l'espace final et qu'on veut revenir
        progressivement vers la source.
        """

        self._check_input_2d(x, name="x")

        if nb_flows < 0 or nb_flows > self.nb_flows:
            raise ValueError(
                f"nb_flows doit être compris entre 0 et {self.nb_flows}."
            )

        device, dtype = self._get_device_dtype()
        x = x.to(device=device, dtype=dtype)

        reversed_flows = list(reversed(self.flows))

        for i, flow in enumerate(reversed_flows[:nb_flows]):
            output = flow.inverse(x)

            if not isinstance(output, (tuple, list)) or len(output) != 2:
                raise ValueError(
                    "Chaque flow.inverse doit retourner exactement : "
                    "(x_inverse, log_det_inverse)."
                )

            x, _ = output

            if not torch.isfinite(x).all():
                raise ValueError(
                    f"NaN ou Inf détecté dans inverse_barycenter au flow numéro {i}."
                )

        return x

    # ========================================================
    # Fonctions compatibles avec des samplers externes
    # ========================================================

    def sample_x(self, y_sampler, nb_samples):
        """
        Échantillonne y puis applique l'inverse pour obtenir x.

        y_sampler doit retourner :
            data, label

        Returns:
            x = T^{-1}(y)
        """

        data, label = y_sampler(nb_samples)

        y = self._to_tensor(data)
        x = self.inverse(y)

        return x

    def sample_y(self, x_sampler, nb_samples):
        """
        Échantillonne x puis applique le modèle direct pour obtenir y.

        x_sampler doit retourner :
            data, label

        Returns:
            y = T(x)
        """

        data, label = x_sampler(nb_samples)

        x = self._to_tensor(data)
        y, _, _ = self.forward(x)

        return y