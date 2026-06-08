import torch
import torch.nn.functional as F


def rand_projections(
    embedding_dim,
    num_samples,
    device=None,
    dtype=torch.float32
):
    """
    Génère des directions aléatoires normalisées sur la sphère unité.

    Chaque projection theta_j vérifie :
        ||theta_j||_2 = 1
    """

    projections = torch.randn(
        num_samples,
        embedding_dim,
        device=device,
        dtype=dtype
    )

    projections = F.normalize(projections, p=2, dim=1)

    return projections


def sliced_wasserstein_distance(
    encoded_samples,
    distribution_samples,
    num_projections=50,
    p=2,
    device=None,
    root=False,
    reduction="none"
):
    """
    Approximation Monte Carlo de la distance Sliced Wasserstein.

    encoded_samples : échantillons générés, de taille (N, d)
    distribution_samples : échantillons cibles, de taille (N, d)

    Si root=False :
        retourne une approximation de SW_p^p.

    Si root=True :
        retourne une approximation de SW_p.

    Pour l'entraînement, il est recommandé d'utiliser :
        root=False
        reduction="mean"
    """

    if device is None:
        device = encoded_samples.device

    encoded_samples = encoded_samples.to(device)
    distribution_samples = distribution_samples.to(device)

    if encoded_samples.dim() != 2:
        raise ValueError("encoded_samples doit être de taille (N, d).")

    if distribution_samples.dim() != 2:
        raise ValueError("distribution_samples doit être de taille (N, d).")

    if encoded_samples.size(1) != distribution_samples.size(1):
        raise ValueError(
            "Les deux distributions doivent avoir la même dimension d."
        )

    if encoded_samples.size(0) != distribution_samples.size(0):
        raise ValueError(
            "Les deux distributions doivent avoir le même nombre d'échantillons N."
        )

    embedding_dim = encoded_samples.size(1)

    projections = rand_projections(
        embedding_dim=embedding_dim,
        num_samples=num_projections,
        device=device,
        dtype=encoded_samples.dtype
    )

    # Projection des échantillons sur les directions aléatoires
    encoded_projections = encoded_samples.matmul(projections.t())
    distribution_projections = distribution_samples.matmul(projections.t())

    # Chaque ligne correspond à une projection
    encoded_projections = encoded_projections.t()
    distribution_projections = distribution_projections.t()

    # Tri des projections pour calculer Wasserstein en dimension 1
    encoded_projections_sorted = torch.sort(
        encoded_projections,
        dim=1
    )[0]

    distribution_projections_sorted = torch.sort(
        distribution_projections,
        dim=1
    )[0]

    diff = encoded_projections_sorted - distribution_projections_sorted

    # Approximation de W_p^p pour chaque projection
    wasserstein_per_projection = torch.abs(diff).pow(p).mean(dim=1)

    if reduction == "none":
        if root:
            return wasserstein_per_projection.pow(1.0 / p)

        return wasserstein_per_projection

    elif reduction == "mean":
        # Approximation de SW_p^p
        sw_power = wasserstein_per_projection.mean()

        if root:
            # Approximation de SW_p
            return sw_power.pow(1.0 / p)

        return sw_power

    else:
        raise ValueError("reduction doit être 'none' ou 'mean'.")