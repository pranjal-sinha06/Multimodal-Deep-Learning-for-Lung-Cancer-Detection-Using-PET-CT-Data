import os, json, csv, time, random, argparse, shutil
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import f1_score, accuracy_score
from torch.amp import autocast, GradScaler
from stage2_dataset import LungSubtypeDataset

BATCH, LR = 64, 1e-4
NUM_WORKERS = 8
HOME_RESULTS = os.path.expanduser("~/stage2_results")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-train-iters", type=int, default=0)
    ap.add_argument("--max-val-iters", type=int, default=0)
    return ap.parse_args()

@torch.no_grad()
def validate(model, loader, device, criterion, max_iters):
    model.eval()
    ys, ps, loss_sum, n = [], [], 0.0, 0
    for it, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        loss_sum += criterion(out, labels).item() * len(labels); n += len(labels)
        ys += labels.cpu().tolist(); ps += out.argmax(1).cpu().tolist()
        if max_iters and it + 1 >= max_iters: break
    return (loss_sum / max(n, 1),
            accuracy_score(ys, ps),
            f1_score(ys, ps, average="macro", zero_division=0),
            f1_score(ys, ps, average=None, labels=[0, 1, 2], zero_division=0))

def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    run_dir = os.path.join("stage2_runs", args.tag); os.makedirs(run_dir, exist_ok=True)
    last_ckpt = os.path.join(run_dir, "last.pth"); metrics_csv = os.path.join(run_dir, "metrics.csv")

    train_ds, val_ds = LungSubtypeDataset("train"), LungSubtypeDataset("val")
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)

    counts = train_ds.df.label_idx.value_counts().sort_index().values
    weights = torch.tensor(counts.sum() / (3 * counts), dtype=torch.float32).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    print("class weights (A,B,G):", weights.cpu().numpy().round(3))

    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, 3)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = GradScaler("cuda")

    start_epoch, best_f1, no_improve, rows = 0, -1.0, 0, []
    if args.resume and os.path.exists(last_ckpt):
        ck = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
        start_epoch, best_f1, no_improve = ck["epoch"] + 1, ck["best_f1"], ck["no_improve"]
        if os.path.exists(metrics_csv): rows = pd.read_csv(metrics_csv).to_dict("records")
        print(f"resumed at epoch {start_epoch}, best_f1={best_f1:.4f}")

    for epoch in range(start_epoch, args.epochs):
        model.train(); t0 = time.time(); tl, n = 0.0, 0
        for it, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            with autocast("cuda"):
                loss = criterion(model(imgs), labels)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tl += loss.item() * len(labels); n += len(labels)
            if args.max_train_iters and it + 1 >= args.max_train_iters: break
        vloss, vacc, vf1, vf1c = validate(model, val_loader, device, criterion, args.max_val_iters)
        dt = time.time() - t0
        row = {"epoch": epoch, "train_loss": tl / max(n, 1), "val_loss": vloss,
               "val_acc": vacc, "val_macro_f1": vf1,
               "f1_A": vf1c[0], "f1_B": vf1c[1], "f1_G": vf1c[2], "time_s": round(dt, 1)}
        rows.append(row)
        with open(metrics_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys())); w.writeheader(); w.writerows(rows)
        print(f"epoch {epoch}: train_loss {row['train_loss']:.4f} | val_acc {vacc:.3f} "
              f"macroF1 {vf1:.3f} | F1 A{vf1c[0]:.2f} B{vf1c[1]:.2f} G{vf1c[2]:.2f} | {dt:.0f}s")

        improved = vf1 > best_f1
        no_improve = 0 if improved else no_improve + 1
        if improved: best_f1 = vf1
        torch.save({"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "best_f1": best_f1, "no_improve": no_improve}, last_ckpt)
        if improved:
            torch.save({"epoch": epoch, "model": model.state_dict(), "val_macro_f1": best_f1},
                       os.path.join(run_dir, "best.pth"))
            print(f"  new best macroF1 {best_f1:.4f} -> best.pth")
        if no_improve >= args.patience:
            print(f"early stopping at epoch {epoch}"); break

    os.makedirs(HOME_RESULTS, exist_ok=True)
    for f in ["best.pth", "metrics.csv"]:
        src = os.path.join(run_dir, f)
        if os.path.exists(src): shutil.copy(src, os.path.join(HOME_RESULTS, f"stage2_{args.tag}_{f}"))
    print(f"done. best val macroF1 {best_f1:.4f}. artifacts in {run_dir} and {HOME_RESULTS}")

if __name__ == "__main__":
    main()
