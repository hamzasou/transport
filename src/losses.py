import torch
import torch.nn.functional as F


def rand_projections(
    embedding_dim,
    num_samples,
    device=None,
    dtype=torch.float32
):
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
    num_projections=50,
    p=2,
    device=None,
    root=False,
    reduction="none"
):
    if device is None:
        device = encoded_samples.device

    if num_projections <= 0:
        raise ValueError("num_projections doit être strictement positif.")

    if p < 1:
        raise ValueError("p doit être supérieur ou égal à 1.")

    if reduction not in ["none", "mean"]:
        raise ValueError("reduction doit être 'none' ou 'mean'.")

    encoded_samples = encoded_samples.to(device)
    distribution_samples = distribution_samples.to(
        device=device,
        dtype=encoded_samples.dtype
    )

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

    if not torch.isfinite(encoded_samples).all():
        raise ValueError("encoded_samples contient NaN ou Inf.")

    if not torch.isfinite(distribution_samples).all():
        raise ValueError("distribution_samples contient NaN ou Inf.")

    embedding_dim = encoded_samples.size(1)

    projections = rand_projections(
        embedding_dim=embedding_dim,
        num_samples=num_projections,
        device=device,
        dtype=encoded_samples.dtype
    )

    encoded_projections = encoded_samples.matmul(projections.t())
    distribution_projections = distribution_samples.matmul(projections.t())

    encoded_projections = encoded_projections.t()
    distribution_projections = distribution_projections.t()

    encoded_projections_sorted = torch.sort(
        encoded_projections,
        dim=1
    )[0]

    distribution_projections_sorted = torch.sort(
        distribution_projections,
        dim=1
    )[0]

    diff = encoded_projections_sorted - distribution_projections_sorted

    wasserstein_per_projection = torch.abs(diff).pow(p).mean(dim=1)

    if reduction == "none":
        if root:
            return wasserstein_per_projection.clamp_min(1e-12).pow(1.0 / p)

        return wasserstein_per_projection

    sw_power = wasserstein_per_projection.mean()

    if root:
        return sw_power.clamp_min(1e-12).pow(1.0 / p)

    return sw_power