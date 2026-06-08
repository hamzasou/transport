import os
import sys
import subprocess
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader

from IPython.display import Image, display, clear_output


# ============================================================
# Seed pour reproductibilité
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Reproductibilité plus stricte
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ============================================================
# Clonage / chemin du projet
# ============================================================

PROJECT_DIR = "/kaggle/working/transport"

if not os.path.exists(PROJECT_DIR):
    print("Clonage du projet...")
    subprocess.run(
        ["git", "clone", "https://github.com/hamzasou/transport.git", PROJECT_DIR],
        check=True
    )
else:
    print("Le projet existe déjà :", PROJECT_DIR)

src_path = os.path.join(PROJECT_DIR, "src")

if os.path.exists(src_path):
    if PROJECT_DIR not in sys.path:
        sys.path.append(PROJECT_DIR)

    print("Chemin ajouté :", PROJECT_DIR)
else:
    raise FileNotFoundError(
        "Le dossier src est introuvable. Vérifie le dépôt GitHub."
    )


# ============================================================
# Imports depuis les fichiers corrigés
# ============================================================

from src.losses import sliced_wasserstein_distance
from src.models import SWFlowModel


# ============================================================
# Chemins Kaggle
# ============================================================

SAVE_DIR = "/kaggle/working/mnist_swot_flow_stable_flows10_bs256_lr1e4_noise07"
OUTDIR = os.path.join(SAVE_DIR, "mnist_results")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)


# ============================================================
# Vérification GPU
# ============================================================

print("Version PyTorch :", torch.__version__)
print("CUDA disponible :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Nombre de GPU :", torch.cuda.device_count())
    print("Nom du GPU :", torch.cuda.get_device_name(0))
else:
    print("Aucun GPU détecté. Active GPU dans Kaggle.")


# ============================================================
# Affichage notebook Kaggle
# ============================================================

def display_image(path):
    display(Image(filename=path))


# ============================================================
# Sécurisation des tenseurs scalaires
# ============================================================

def to_scalar(x, name):
    """
    Convertit un tenseur en scalaire si nécessaire.

    Cette fonction évite les erreurs lorsque cost ou shatten
    retournent un vecteur au lieu d'un scalaire.
    """

    if not torch.is_tensor(x):
        raise TypeError(f"{name} doit être un torch.Tensor.")

    if x.ndim > 0:
        x = x.mean()

    if not torch.isfinite(x):
        raise ValueError(f"{name} contient NaN ou Inf.")

    return x


# ============================================================
# MLP utilisé dans RealNVP
# ============================================================

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=512):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Couche RealNVP stable
# ============================================================

class StableRealNVP(nn.Module):
    def __init__(self, dim=784, hidden_dim=512):
        super().__init__()

        if dim % 2 != 0:
            raise ValueError(
                "dim doit être pair pour cette implémentation RealNVP."
            )

        self.dim = dim
        half = dim // 2

        self.t1 = MLP(half, half, hidden_dim)
        self.s1 = MLP(half, half, hidden_dim)

        self.t2 = MLP(half, half, hidden_dim)
        self.s2 = MLP(half, half, hidden_dim)

    def forward(self, x):
        lower = x[:, :self.dim // 2]
        upper = x[:, self.dim // 2:]

        s1 = 1.5 * torch.tanh(self.s1(lower))
        t1 = self.t1(lower)

        upper = upper * torch.exp(s1) + t1

        s2 = 1.5 * torch.tanh(self.s2(upper))
        t2 = self.t2(upper)

        lower = lower * torch.exp(s2) + t2

        z = torch.cat([lower, upper], dim=1)

        log_diag = torch.cat([s1, s2], dim=1)
        log_det = torch.sum(s1, dim=1) + torch.sum(s2, dim=1)

        return z, log_diag, log_det

    def inverse(self, z):
        lower = z[:, :self.dim // 2]
        upper = z[:, self.dim // 2:]

        s2 = 1.5 * torch.tanh(self.s2(upper))
        t2 = self.t2(upper)

        lower = (lower - t2) * torch.exp(-s2)

        s1 = 1.5 * torch.tanh(self.s1(lower))
        t1 = self.t1(lower)

        upper = (upper - t1) * torch.exp(-s1)

        x = torch.cat([lower, upper], dim=1)

        log_det = -torch.sum(s1, dim=1) - torch.sum(s2, dim=1)

        return x, log_det


# ============================================================
# Test rapide de l'inversibilité RealNVP
# ============================================================

def test_realnvp_inverse(device, dim=784, hidden_dim=512):
    """
    Vérifie que inverse(forward(x)) ≈ x.
    Ce test ne remplace pas l'entraînement, mais il vérifie
    la cohérence mathématique de la couche RealNVP.
    """

    layer = StableRealNVP(dim=dim, hidden_dim=hidden_dim).to(device)
    layer.eval()

    with torch.no_grad():
        x = torch.randn(8, dim, device=device)

        z, _, _ = layer(x)
        x_rec, _ = layer.inverse(z)

        error = torch.max(torch.abs(x - x_rec)).item()

    print("Test inverse RealNVP | erreur max :", error)

    if error > 1e-4:
        print("Attention : erreur d'inversion élevée.")
    else:
        print("Test inverse RealNVP validé.")


# ============================================================
# Sauvegarde images générées
# ============================================================

def save_generated_images(
    model,
    epoch,
    fixed_noise,
    outdir=OUTDIR,
    display_result=True
):
    model.eval()
    os.makedirs(outdir, exist_ok=True)

    with torch.no_grad():
        generated, _, _ = model(fixed_noise)

        print(
            "Stats generated avant affichage | "
            f"min={generated.min().item():.4f}, "
            f"max={generated.max().item():.4f}, "
            f"mean={generated.mean().item():.4f}, "
            f"std={generated.std().item():.4f}"
        )

        generated = generated.view(fixed_noise.size(0), 1, 28, 28)

        # Les images MNIST réelles sont normalisées dans [-1, 1].
        # Donc on ramène les sorties dans [0, 1] seulement pour l'affichage.
        generated = (generated + 1) / 2
        generated = torch.clamp(generated, 0, 1)

        path = os.path.join(outdir, f"generated_epoch_{epoch}.png")

        utils.save_image(
            generated,
            path,
            nrow=8
        )

    model.train()

    if display_result:
        print(f"Images générées - epoch {epoch}")
        display_image(path)

    return path


# ============================================================
# Sauvegarde comparaison réel / généré
# ============================================================

def save_comparison_images(
    model,
    real_images,
    epoch,
    fixed_noise,
    outdir=OUTDIR,
    display_result=True
):
    model.eval()
    os.makedirs(outdir, exist_ok=True)

    n = fixed_noise.size(0)

    with torch.no_grad():
        device = fixed_noise.device

        real_images = real_images[:n].to(device)

        generated, _, _ = model(fixed_noise)
        generated = generated.view(n, 1, 28, 28)

        real_images = (real_images + 1) / 2
        generated = (generated + 1) / 2

        real_images = torch.clamp(real_images, 0, 1)
        generated = torch.clamp(generated, 0, 1)

        comparison = torch.cat([real_images, generated], dim=0)

        path = os.path.join(outdir, f"comparison_epoch_{epoch}.png")

        utils.save_image(
            comparison,
            path,
            nrow=8
        )

    model.train()

    if display_result:
        print(f"Comparaison MNIST réel / généré - epoch {epoch}")
        print("Premières lignes : vraies images MNIST")
        print("Dernières lignes : images générées")
        display_image(path)

    return path


# ============================================================
# Courbe loss
# ============================================================

def save_loss_curve(
    loss_history,
    sw_history,
    epoch,
    outdir=OUTDIR,
    display_result=True
):
    os.makedirs(outdir, exist_ok=True)

    path = os.path.join(outdir, f"loss_curve_epoch_{epoch}.png")

    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, label="Loss totale")
    plt.plot(sw_history, label="Sliced Wasserstein SW2^2")
    plt.xlabel("Epoch")
    plt.ylabel("Valeur")
    plt.title("Évolution de la loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    if display_result:
        print(f"Courbe de loss - epoch {epoch}")
        display_image(path)

    return path


# ============================================================
# Sauvegarde checkpoint
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss_history,
    sw_history,
    path
):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss_history": loss_history,
        "sw_history": sw_history,
        "noise_std": model.noise_std if hasattr(model, "noise_std") else None,
    }, path)

    print("Checkpoint sauvegardé :", path)


# ============================================================
# Chargement checkpoint
# ============================================================

def load_checkpoint(model, optimizer, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    loss_history = checkpoint.get("loss_history", [])
    sw_history = checkpoint.get("sw_history", [])

    if "noise_std" in checkpoint and checkpoint["noise_std"] is not None:
        if hasattr(model, "set_noise_std"):
            model.set_noise_std(checkpoint["noise_std"])

    print("Checkpoint chargé :", checkpoint_path)
    print("Reprise à partir de epoch :", start_epoch)

    return model, optimizer, start_epoch, loss_history, sw_history


# ============================================================
# Entraînement principal
# ============================================================

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device utilisé :", device)

    # ========================================================
    # Hyperparamètres recommandés
    # ========================================================
        
    
   
    batch_size = 256
    epochs = 400
    
    dim = 28 * 28

    # Version recommandée stable
    nb_flows = 14

    # Si tu veux une version plus lourde après validation :
    # nb_flows = 14

    hidden_dim = 512

    # Learning rate plus adapté que 3e-5 pour recommencer depuis zéro
    lr = 4e-5

    # 500 est beaucoup moins lourd que 1000 et reste correct
    num_projections = 2000
    lamb = 1.8e-4
    gamma = 1.2
 

    noise_std = 0.5

    display_every = 10
    checkpoint_every = 25

    RESUME = False
    RESUME_CHECKPOINT_PATH = ""

    # ========================================================
    # Dataset MNIST
    # ========================================================

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.MNIST(
        root="/kaggle/working/data",
        train=True,
        download=True,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # ========================================================
    # Test de la couche RealNVP
    # ========================================================

    test_realnvp_inverse(
        device=device,
        dim=dim,
        hidden_dim=hidden_dim
    )

    # ========================================================
    # Modèle SWOT-Flow
    # ========================================================

    flows = [
        StableRealNVP(dim=dim, hidden_dim=hidden_dim)
        for _ in range(nb_flows)
    ]

    model = SWFlowModel(
        flows=flows,
        device=device,
        noise_std=noise_std
    ).to(device)

    total_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("Nombre de paramètres entraînables :", total_params)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_epoch = 1
    loss_history = []
    sw_history = []

    best_loss = float("inf")
    best_sw = float("inf")

    # ========================================================
    # Reprendre un entraînement si activé
    # ========================================================

    if RESUME:
        model, optimizer, start_epoch, loss_history, sw_history = load_checkpoint(
            model=model,
            optimizer=optimizer,
            checkpoint_path=RESUME_CHECKPOINT_PATH,
            device=device
        )

        # On force le learning rate choisi même après chargement
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        best_loss = min(loss_history) if len(loss_history) > 0 else float("inf")
        best_sw = min(sw_history) if len(sw_history) > 0 else float("inf")

    fixed_noise = model.sample_noise(64, dim).to(device)

    print("Start MNIST SWOT-Flow training")
    print("________________________________")
    print("Hyperparamètres :")
    print("batch_size =", batch_size)
    print("epochs =", epochs)
    print("dim =", dim)
    print("nb_flows =", nb_flows)
    print("hidden_dim =", hidden_dim)
    print("lr =", lr)
    print("num_projections =", num_projections)
    print("lambda =", lamb)
    print("gamma =", gamma)
    print("noise_std =", noise_std)
    print("SAVE_DIR =", SAVE_DIR)
    print("OUTDIR =", OUTDIR)
    print("RESUME =", RESUME)
    print("________________________________")

    last_real_images = None

    # ========================================================
    # Boucle d'entraînement
    # ========================================================

    for epoch in range(start_epoch, epochs + 1):

        total_loss = 0.0
        total_sw = 0.0
        total_cost = 0.0
        total_reg = 0.0

        model.train()

        for images, _ in loader:
            images = images.to(device)
            last_real_images = images

            current_batch_size = images.size(0)

            target = images.reshape(current_batch_size, -1)

            source = model.sample_noise(current_batch_size, dim).to(device)

            optimizer.zero_grad(set_to_none=True)

            generated, shatten, _ = model(source)

            sw = sliced_wasserstein_distance(
                generated,
                target,
                num_projections=num_projections,
                p=2,
                device=device,
                root=False,
                reduction="mean"
            )

            cost = model.transport_cost(source)

            sw = to_scalar(sw, "sw")
            cost = to_scalar(cost, "cost")
            shatten_reg = to_scalar(shatten, "shatten_reg")

            loss = sw + lamb * cost + gamma * shatten_reg

            if not torch.isfinite(loss):
                print("NaN ou Inf détecté, arrêt de l'entraînement.")
                print("sw =", sw.item())
                print("cost =", cost.item())
                print("shatten_reg =", shatten_reg.item())
                return

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=10.0
            )

            optimizer.step()

            total_loss += loss.item()
            total_sw += sw.item()
            total_cost += cost.item()
            total_reg += shatten_reg.item()

        avg_loss = total_loss / len(loader)
        avg_sw = total_sw / len(loader)
        avg_cost = total_cost / len(loader)
        avg_reg = total_reg / len(loader)

        loss_history.append(avg_loss)
        sw_history.append(avg_sw)

        cost_contrib = lamb * avg_cost
        reg_contrib = gamma * avg_reg

        if avg_sw > 0:
            cost_percent = 100 * cost_contrib / avg_sw
            reg_percent = 100 * reg_contrib / avg_sw
        else:
            cost_percent = 0.0
            reg_percent = 0.0

        print(
            f"Epoch {epoch}/{epochs} | "
            f"SW2^2: {avg_sw:.6f} | "
            f"Cost: {avg_cost:.6f} | "
            f"Reg: {avg_reg:.6f} | "
            f"Loss: {avg_loss:.6f}"
        )

        print(
            f"Contributions | "
            f"lambda*Cost: {cost_contrib:.6f} ({cost_percent:.2f}% de SW2^2) | "
            f"gamma*Reg: {reg_contrib:.6f} ({reg_percent:.2f}% de SW2^2)"
        )

        # ====================================================
        # Sauvegarde meilleur checkpoint selon loss totale
        # ====================================================

        if avg_loss < best_loss:
            best_loss = avg_loss

            best_checkpoint_path = os.path.join(
                SAVE_DIR,
                "best_checkpoint.pth"
            )

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss_history=loss_history,
                sw_history=sw_history,
                path=best_checkpoint_path
            )

            print("Nouveau meilleur checkpoint selon loss totale sauvegardé.")
            print("Best loss :", best_loss)

        # ====================================================
        # Sauvegarde meilleur checkpoint selon SW seulement
        # ====================================================

        if avg_sw < best_sw:
            best_sw = avg_sw

            best_sw_checkpoint_path = os.path.join(
                SAVE_DIR,
                "best_sw_checkpoint.pth"
            )

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss_history=loss_history,
                sw_history=sw_history,
                path=best_sw_checkpoint_path
            )

            print("Nouveau meilleur checkpoint selon SW sauvegardé.")
            print("Best SW :", best_sw)

        # ====================================================
        # Sauvegardes périodiques
        # ====================================================

        if epoch % checkpoint_every == 0:
            latest_checkpoint_path = os.path.join(
                SAVE_DIR,
                "latest_checkpoint.pth"
            )

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss_history=loss_history,
                sw_history=sw_history,
                path=latest_checkpoint_path
            )

            checkpoint_path = os.path.join(
                SAVE_DIR,
                f"checkpoint_epoch_{epoch}.pth"
            )

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss_history=loss_history,
                sw_history=sw_history,
                path=checkpoint_path
            )

        # ====================================================
        # Affichage périodique
        # ====================================================

        if epoch % display_every == 0:
            clear_output(wait=True)

            print(
                f"Epoch {epoch}/{epochs} | "
                f"SW2^2: {avg_sw:.6f} | "
                f"Cost: {avg_cost:.6f} | "
                f"Reg: {avg_reg:.6f} | "
                f"Loss: {avg_loss:.6f}"
            )

            print(
                f"Contributions | "
                f"lambda*Cost: {cost_contrib:.6f} ({cost_percent:.2f}% de SW2^2) | "
                f"gamma*Reg: {reg_contrib:.6f} ({reg_percent:.2f}% de SW2^2)"
            )

            save_generated_images(
                model=model,
                epoch=epoch,
                fixed_noise=fixed_noise,
                outdir=OUTDIR,
                display_result=True
            )

            save_comparison_images(
                model=model,
                real_images=last_real_images,
                epoch=epoch,
                fixed_noise=fixed_noise,
                outdir=OUTDIR,
                display_result=True
            )

            save_loss_curve(
                loss_history=loss_history,
                sw_history=sw_history,
                epoch=epoch,
                outdir=OUTDIR,
                display_result=True
            )

    # ========================================================
    # Sauvegarde finale
    # ========================================================

    print("Génération finale après entraînement...")

    save_generated_images(
        model=model,
        epoch="final",
        fixed_noise=fixed_noise,
        outdir=OUTDIR,
        display_result=True
    )

    save_comparison_images(
        model=model,
        real_images=last_real_images,
        epoch="final",
        fixed_noise=fixed_noise,
        outdir=OUTDIR,
        display_result=True
    )

    save_loss_curve(
        loss_history=loss_history,
        sw_history=sw_history,
        epoch="final",
        outdir=OUTDIR,
        display_result=True
    )

    model_path = os.path.join(
        SAVE_DIR,
        "mnist_swot_flow_stable_final.pth"
    )

    torch.save(model.state_dict(), model_path)

    print("Modèle final sauvegardé :", model_path)

    final_checkpoint_path = os.path.join(
        SAVE_DIR,
        "final_checkpoint.pth"
    )

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epochs,
        loss_history=loss_history,
        sw_history=sw_history,
        path=final_checkpoint_path
    )

    print("Tous les résultats sont sauvegardés dans :", SAVE_DIR)


if __name__ == "__main__":
    main()