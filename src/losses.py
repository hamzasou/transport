import torch
import torch.nn.functional as F


def rand_projections(
    embedding_dim,
    num_samples,
    device=None,
    dtype=torch.float32,
    positive_directions=False
):
    """
    Génère des directions aléatoires normalisées.

    Si positive_directions=False :
        on génère des directions gaussiennes normalisées.
        C'est le choix standard pour la Sliced Wasserstein.

    Si positive_directions=True :
        on génère des directions positives normalisées.
        C'est une variante expérimentale adaptée aux données dans [0,1]^d.
    """

    if positive_directions:
        projections = torch.rand(
            num_samples,
            embedding_dim,
            device=device,
            dtype=dtype
        )
    else:
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
    reduction="none",
    positive_directions=False
):
    """
    Approximation Monte Carlo de la distance Sliced Wasserstein.

    Paramètres
    ----------
    encoded_samples : torch.Tensor
        Échantillons générés par le modèle, de taille (N, d).

    distribution_samples : torch.Tensor
        Échantillons de la distribution cible, de taille (N, d).

    num_projections : int
        Nombre de projections aléatoires utilisées.

    p : int ou float
        Ordre de la distance de Wasserstein.

    device : torch.device
        Device utilisé : CPU ou GPU.

    root : bool
        Si root=False :
            retourne une approximation de SW_p^p.

        Si root=True :
            retourne une approximation de SW_p.

    reduction : str
        "none" :
            retourne la valeur pour chaque projection.

        "mean" :
            retourne la moyenne sur toutes les projections.

    positive_directions : bool
        Si False :
            directions gaussiennes normalisées, version standard.

        Si True :
            directions positives normalisées, variante expérimentale.

    Remarque
    --------
    Pour l'entraînement SWOT-Flow, il est recommandé d'utiliser :
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
        dtype=encoded_samples.dtype,
        positive_directions=positive_directions
    )

    encoded_projections = encoded_samples.matmul(projections.t())
    distribution_projections = distribution_samples.matmul(projections.t())

    encoded_projections = encoded_projections.t()
    distribution_projections = distribution_projections.t()

    encoded_projections_sorted = torch.sort(encoded_projections, dim=1)[0]
    distribution_projections_sorted = torch.sort(distribution_projections, dim=1)[0]

    diff = encoded_projections_sorted - distribution_projections_sorted

    wasserstein_per_projection = torch.abs(diff).pow(p).mean(dim=1)

    if reduction == "none":
        if root:
            return wasserstein_per_projection.pow(1.0 / p)
        else:
            return wasserstein_per_projection

    elif reduction == "mean":
        sw = wasserstein_per_projection.mean()

        if root:
            sw = sw.pow(1.0 / p)

        return sw

    else:
        raise ValueError("reduction doit être 'none' ou 'mean'.")