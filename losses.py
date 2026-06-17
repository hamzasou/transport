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
    """

    if embedding_dim <= 0:
        raise ValueError("embedding_dim doit être strictement positif.")

    if num_samples <= 0:
        raise ValueError("num_samples doit être strictement positif.")

    projections = torch.randn(
        num_samples,
        embedding_dim,
        device=device,
        dtype=dtype
    )

    projections = F.normalize(
        projections,
        p=2,
        dim=1,
        eps=1e-12
    )

    return projections


def sliced_wasserstein_distance(
    encoded_samples,
    distribution_samples,
    num_projections=300,
    p=2,
    device=None,
    root=False,
    reduction="mean"
):
    """
    Approximation Monte Carlo de la distance Sliced-Wasserstein.

    encoded_samples : tenseur de taille (N, d)
    distribution_samples : tenseur de taille (N, d)

    Si root=False :
        retourne SW_p^p

    Si root=True :
        retourne SW_p

    Pour l'entraînement :
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
        raise ValueError("Les deux distributions doivent avoir la même dimension d.")

    if encoded_samples.size(0) != distribution_samples.size(0):
        raise ValueError("Les deux distributions doivent avoir le même nombre d'échantillons N.")

    if p < 1:
        raise ValueError("p doit être supérieur ou égal à 1.")

    embedding_dim = encoded_samples.size(1)

    projections = rand_projections(
        embedding_dim=embedding_dim,
        num_samples=num_projections,
        device=device,
        dtype=encoded_samples.dtype
    )

    # Projections : (N, d) @ (d, J) = (N, J)
    encoded_projections = encoded_samples.matmul(projections.t())
    distribution_projections = distribution_samples.matmul(projections.t())

    # Tri selon les échantillons pour chaque projection
    encoded_sorted = torch.sort(encoded_projections, dim=0)[0]
    distribution_sorted = torch.sort(distribution_projections, dim=0)[0]

    diff = encoded_sorted - distribution_sorted

    # W_p^p pour chaque projection
    wasserstein_p_per_projection = torch.abs(diff).pow(p).mean(dim=0)

    if reduction == "none":
        if root:
            return wasserstein_p_per_projection.pow(1.0 / p)
        else:
            return wasserstein_p_per_projection

    elif reduction == "mean":
        sw_p = wasserstein_p_per_projection.mean()

        if root:
            return sw_p.pow(1.0 / p)
        else:
            return sw_p

    else:
        raise ValueError("reduction doit être 'none' ou 'mean'.")