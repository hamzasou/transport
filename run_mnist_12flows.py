import os
import sys
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader

from IPython.display import Image, display, clear_output


# ============================================================
# Projet
# ============================================================

PROJECT_DIR = "/kaggle/working/transport"

if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

from src.losses import sliced_wasserstein_distance
from src.models import SWFlowModel, StableRealNVP


# ============================================================
# Seed
# ============================================================

torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# ============================================================
# Chemins
# ============================================================

SAVE_DIR = "/kaggle/working/mnist_swot_flow_light_realnvp"
OUTDIR = os.path.join(SAVE_DIR, "mnist_results")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)


# ============================================================
# Affichage
# ============================================================

def display_image(path):
    display(Image(filename=path))


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
# Comparaison réel / généré
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
    device = fixed_noise.device

    with torch.no_grad():
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
    plt.plot(sw_history, label="Sliced Wasserstein SW2")
    plt.xlabel("Epoch")
    plt.ylabel("Valeur")
    plt.title("Évolution de la loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    plt.close()

    if display_result:
        print(f"Courbe de loss - epoch {epoch}")
        display_image(path)

    return path


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss_history,
    sw_history,
    path
):
    tmp_path = path + ".tmp"

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss_history": loss_history,
        "sw_history": sw_history,
    }, tmp_path)

    os.replace(tmp_path, path)

    print("Checkpoint sauvegardé :", path)


def load_checkpoint(model, optimizer, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    loss_history = checkpoint["loss_history"]
    sw_history = checkpoint["sw_history"]

    print("Checkpoint chargé :", checkpoint_path)
    print("Reprise à partir de epoch :", start_epoch)

    return model, optimizer, start_epoch, loss_history, sw_history


# ============================================================
# Entraînement
# ============================================================

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Version PyTorch :", torch.__version__)
    print("CUDA disponible :", torch.cuda.is_available())
    print("Device utilisé :", device)

    if torch.cuda.is_available():
        print("Nombre de GPU :", torch.cuda.device_count())
        print("Nom du GPU :", torch.cuda.get_device_name(0))
    else:
        print("Aucun GPU détecté. Active GPU dans Kaggle.")

    # ========================================================
    # Hyperparamètres légers
    # ========================================================
    
    batch_size = 128
    epochs = 400

    dim = 28 * 28

    nb_flows = 12
    hidden_dim = 256
    num_res_blocks = 2
    scale_clip = 1.0
    
    lr = 2e-4
    num_projections = 1000

    lamb = 7e-3
    gamma = 7e-3

    display_every = 5
    checkpoint_every = 25
    save_latest_every = 5

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
    # Modèle
    # ========================================================

    flows = [
        StableRealNVP(
            dim=dim,
            hidden_dim=hidden_dim,
            num_res_blocks=num_res_blocks,
            scale_clip=scale_clip,
            use_random_permutation=True
        )
        for _ in range(nb_flows)
    ]

    model = SWFlowModel(flows, device=device).to(device)

    total_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("Nombre de paramètres entraînables :", total_params)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-6
    )

    start_epoch = 1
    loss_history = []
    sw_history = []

    best_loss = float("inf")

    # ========================================================
    # Reprise checkpoint
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
    print("batch_size =", batch_size)
    print("epochs =", epochs)
    print("dim =", dim)
    print("nb_flows =", nb_flows)
    print("hidden_dim =", hidden_dim)
    print("num_res_blocks =", num_res_blocks)
    print("scale_clip =", scale_clip)
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

        model.train()

        for images, _ in loader:
            images = images.to(device)
            last_real_images = images

            current_batch_size = images.size(0)

            target = images.view(current_batch_size, -1)

            source = torch.randn(
                current_batch_size,
                dim,
                device=device
            )

            optimizer.zero_grad(set_to_none=True)

            generated, shatten_reg, _, cost = model(
                source,
                return_cost=True
            )

            sw = sliced_wasserstein_distance(
                generated,
                target,
                num_projections=num_projections,
                p=2,
                device=device,
                root=False,
                reduction="mean"
            )

            loss = sw + lamb * cost + gamma * shatten_reg

            if torch.isnan(loss) or torch.isinf(loss):
                print("NaN ou Inf détecté, arrêt de l'entraînement")
                return

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0
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
            f"SW2: {avg_sw:.6f} | "
            f"Cost: {avg_cost:.6f} | "
            f"Reg: {avg_reg:.6f} | "
            f"Loss: {avg_loss:.6f}"
        )

        print(
            f"Contributions | "
            f"lambda*Cost: {cost_contrib:.8f} ({cost_percent:.2f}% de SW) | "
            f"gamma*Reg: {reg_contrib:.8f} ({reg_percent:.2f}% de SW)"
        )

        # Sauvegarde meilleur modèle
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

            print("Nouveau meilleur checkpoint sauvegardé.")
            print("Best loss :", best_loss)

        # Sauvegarde latest moins fréquente
        if epoch % save_latest_every == 0:
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

        # Checkpoint périodique
        if epoch % checkpoint_every == 0:
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

        # Affichage
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
                f"lambda*Cost: {cost_contrib:.8f} ({cost_percent:.2f}% de SW) | "
                f"gamma*Reg: {reg_contrib:.8f} ({reg_percent:.2f}% de SW)"
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
        f"mnist_swot_flow_flows{nb_flows}_bs{batch_size}_light_realnvp.pth"
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