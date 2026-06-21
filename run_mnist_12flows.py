import os
import sys
import subprocess
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

torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True


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
    sys.path.append(PROJECT_DIR)
    print("Chemin ajouté :", PROJECT_DIR)
else:
    raise FileNotFoundError("Le dossier src est introuvable. Vérifie le dépôt GitHub.")

from src.losses import sliced_wasserstein_distance
from src.models import SWFlowModel


# ============================================================
# Chemins Kaggle
# ============================================================

SAVE_DIR = "/kaggle/working/mnist_swot_flow_flows14_bs256_lr5e5_lamb85e7_gamma33e9"
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

    for i in range(torch.cuda.device_count()):
        print(f"GPU {i} :", torch.cuda.get_device_name(i))
else:
    print("Aucun GPU détecté. Active GPU dans Kaggle.")


# ============================================================
# Fonctions utilitaires pour DataParallel
# ============================================================

def unwrap_model(model):
    """
    Si le modèle est enveloppé par nn.DataParallel,
    retourne le modèle interne réel.
    Sinon, retourne le modèle lui-même.
    """
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def get_core_model(model):
    """
    Retourne le vrai modèle SWFlowModel, même si on utilise :
    - nn.DataParallel
    - SWFlowModelWithCost
    """
    base_model = unwrap_model(model)

    if hasattr(base_model, "swflow"):
        return base_model.swflow

    return base_model


def remove_module_prefix(state_dict):
    """
    Corrige les checkpoints sauvegardés avec DataParallel,
    où les clés commencent parfois par 'module.'.
    """
    new_state_dict = {}

    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[len("module."):]] = v
        else:
            new_state_dict[k] = v

    return new_state_dict


# ============================================================
# Affichage notebook Kaggle
# ============================================================

def display_image(path):
    display(Image(filename=path))


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
# Wrapper pour paralléliser aussi transport_cost
# ============================================================

class SWFlowModelWithCost(nn.Module):
    """
    Wrapper autour de SWFlowModel.

    Objectif :
    faire passer generated, shatten, log_det et transport_cost
    dans le forward afin que nn.DataParallel répartisse tout le calcul
    sur les GPU disponibles.
    """

    def __init__(self, flows, device):
        super().__init__()
        self.swflow = SWFlowModel(flows, device)

    def forward(self, source):
        generated, shatten, log_det = self.swflow(source)
        cost = self.swflow.transport_cost(source)

        return generated, shatten, log_det, cost


# ============================================================
# Sauvegarde images générées
# ============================================================

def save_generated_images(
    model,
    device,
    epoch,
    fixed_noise,
    outdir=OUTDIR,
    display_result=True
):
    model.eval()
    os.makedirs(outdir, exist_ok=True)

    with torch.no_grad():
        generated = model(fixed_noise)[0]

        generated = generated.view(fixed_noise.size(0), 1, 28, 28)

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
    device,
    epoch,
    fixed_noise,
    outdir=OUTDIR,
    display_result=True
):
    model.eval()
    os.makedirs(outdir, exist_ok=True)

    n = fixed_noise.size(0)

    with torch.no_grad():
        real_images = real_images[:n].to(device)

        generated = model(fixed_noise)[0]
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
    plt.plot(sw_history, label="Sliced Wasserstein SW2")
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
# Sauvegarde checkpoint compatible DataParallel
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss_history,
    sw_history,
    path
):
    core_model = get_core_model(model)

    torch.save({
        "epoch": epoch,
        "model_state_dict": core_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss_history": loss_history,
        "sw_history": sw_history,
    }, path)

    print("Checkpoint sauvegardé :", path)


# ============================================================
# Chargement checkpoint compatible DataParallel
# ============================================================

def load_checkpoint(model, optimizer, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = checkpoint["model_state_dict"]
    state_dict = remove_module_prefix(state_dict)

    core_model = get_core_model(model)
    core_model.load_state_dict(state_dict)

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    loss_history = checkpoint["loss_history"]
    sw_history = checkpoint["sw_history"]

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
    # Hyperparamètres
    # ========================================================

    batch_size = 256
    epochs = 400

    dim = 28 * 28
    nb_flows = 14
    hidden_dim = 512

    lr = 3e-5
    num_projections = 2500

    lamb = 8.5e-6
    gamma = 3.3e-8

    display_every = 10
    checkpoint_every = 10

    RESUME = False
    RESUME_CHECKPOINT_PATH = ""

    fixed_noise = torch.randn(64, dim, device=device)

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
    # Modèle SWOT-Flow
    # ========================================================

    flows = [
        StableRealNVP(dim=dim, hidden_dim=hidden_dim)
        for _ in range(nb_flows)
    ]

    model = SWFlowModelWithCost(flows, device).to(device)

    # ========================================================
    # Parallélisme multi-GPU
    # ========================================================

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Utilisation de {torch.cuda.device_count()} GPU avec nn.DataParallel")
        model = nn.DataParallel(model)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Nombre de paramètres entraînables :", total_params)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_epoch = 1
    loss_history = []
    sw_history = []

    best_loss = float("inf")

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

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        best_loss = min(loss_history) if len(loss_history) > 0 else float("inf")

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
    print("SAVE_DIR =", SAVE_DIR)
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

        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            last_real_images = images

            current_batch_size = images.size(0)

            target = images.view(current_batch_size, -1)

            source = torch.randn(current_batch_size, dim, device=device)

            optimizer.zero_grad()

            # ------------------------------------------------
            # Passage avant principal.
            # Avec DataParallel, generated, shatten et cost
            # sont répartis sur les GPU.
            # ------------------------------------------------

            generated, shatten, _, cost = model(source)

            sw = sliced_wasserstein_distance(
                generated,
                target,
                num_projections=num_projections,
                p=2,
                device=device,
                root=False,
                reduction="mean"
            )

            cost = cost.mean()

            if torch.is_tensor(shatten):
                shatten_reg = shatten.mean()
            else:
                shatten_reg = shatten

            loss = sw + lamb * cost + gamma * shatten_reg

            if torch.isnan(loss):
                print("NaN détecté, arrêt de l'entraînement")
                return

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_sw += sw.item()
            total_cost += cost.item()

            if torch.is_tensor(shatten_reg):
                total_reg += shatten_reg.item()
            else:
                total_reg += float(shatten_reg)

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
            f"SW2: {avg_sw:.6f} | "
            f"Cost: {avg_cost:.6f} | "
            f"Reg: {avg_reg:.6f} | "
            f"Loss: {avg_loss:.6f}"
        )

        print(
            f"Contributions | "
            f"lambda*Cost: {cost_contrib:.6f} ({cost_percent:.2f}% de SW) | "
            f"gamma*Reg: {reg_contrib:.6f} ({reg_percent:.2f}% de SW)"
        )

        # ----------------------------------------------------
        # Checkpoint latest
        # ----------------------------------------------------

        latest_checkpoint_path = os.path.join(SAVE_DIR, "latest_checkpoint.pth")

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss_history=loss_history,
            sw_history=sw_history,
            path=latest_checkpoint_path
        )

        # ----------------------------------------------------
        # Checkpoint best
        # ----------------------------------------------------

        if avg_loss < best_loss:
            best_loss = avg_loss

            best_checkpoint_path = os.path.join(SAVE_DIR, "best_checkpoint.pth")

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss_history=loss_history,
                sw_history=sw_history,
                path=best_checkpoint_path
            )

            print("Nouveau meilleur checkpoint sauvegardé.")
            print("Best loss :", best_loss)

        # ----------------------------------------------------
        # Checkpoint par epoch
        # ----------------------------------------------------

        if epoch % checkpoint_every == 0:
            checkpoint_path = os.path.join(SAVE_DIR, f"checkpoint_epoch_{epoch}.pth")

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss_history=loss_history,
                sw_history=sw_history,
                path=checkpoint_path
            )

        # ----------------------------------------------------
        # Affichage et sauvegarde des figures
        # ----------------------------------------------------

        if epoch % display_every == 0:
            clear_output(wait=True)

            print(
                f"Epoch {epoch}/{epochs} | "
                f"SW2: {avg_sw:.6f} | "
                f"Cost: {avg_cost:.6f} | "
                f"Reg: {avg_reg:.6f} | "
                f"Loss: {avg_loss:.6f}"
            )

            print(
                f"Contributions | "
                f"lambda*Cost: {cost_contrib:.6f} ({cost_percent:.2f}% de SW) | "
                f"gamma*Reg: {reg_contrib:.6f} ({reg_percent:.2f}% de SW)"
            )

            save_generated_images(
                model=model,
                device=device,
                epoch=epoch,
                fixed_noise=fixed_noise,
                outdir=OUTDIR,
                display_result=True
            )

            save_comparison_images(
                model=model,
                real_images=last_real_images,
                device=device,
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
    # Génération finale après entraînement
    # ========================================================

    print("Génération finale après entraînement...")

    save_generated_images(
        model=model,
        device=device,
        epoch="final",
        fixed_noise=fixed_noise,
        outdir=OUTDIR,
        display_result=True
    )

    save_comparison_images(
        model=model,
        real_images=last_real_images,
        device=device,
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

    # ========================================================
    # Sauvegarde finale du modèle seul
    # ========================================================

    model_path = os.path.join(
        SAVE_DIR,
        "mnist_swot_flow_flows14_bs256_lr5e5_balanced.pth"
    )

    torch.save(get_core_model(model).state_dict(), model_path)

    print("Modèle final sauvegardé :", model_path)

    # ========================================================
    # Sauvegarde finale complète
    # ========================================================

    final_checkpoint_path = os.path.join(SAVE_DIR, "final_checkpoint.pth")

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epochs,
        loss_history=loss_history,
        sw_history=sw_history,
        path=final_checkpoint_path
    )

    print("Tous les résultats sont sauvegardés dans :", SAVE_DIR)


# ============================================================
# Lancer l'entraînement
# ============================================================

if __name__ == "__main__":
    main()