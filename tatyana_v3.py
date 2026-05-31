"""
Tatyana v3 — Residual MLP surrogate for GENE linear stability~
Extends v2 with: 相比于V2的升级
  [1] Species-resolved linear outputs as regression targets
      (gamma_e, omega_e, gamma_i, omega_i) — proxies for SAT3 WeL/WiL
      R = |gamma_e / gamma_i| feeds the future mode-switching M(R) in Item 3
  [2] kmax & gamma_max obtained analytically at inference time by scanning
      the main head over a ky grid — no separate global head

Mapping: (kymin, trpeps, shat, q0, omt_i, omt_e, omn)
      -> (gamma, omega, gamma_e, omega_e, gamma_i, omega_i)

Author : Tingyi Chen
Email  : flyawaypencil480@gmail.com
Date   : 2026-05-31
"""

import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

# Configuration
DATA_PATH   = Path("df_clean_reconstructed.tsv") # Please replace it with your dataset name !!
CKPT_PATH   = Path("tatyana_v3.pt")
SCALER_PATH = Path("tatyana_v3_scalers.pkl")

FEATURES     = ["kymin", "trpeps", "shat", "q0", "omt_i", "omt_e", "omn"]
EQ_FEATURES  = ["trpeps", "shat", "q0", "omt_i", "omt_e", "omn"]
BASE_TARGETS = ["gamma", "omega"]
WEIGHT_COLS  = ["gamma_e", "omega_e", "gamma_i", "omega_i"]
ALL_TARGETS  = BASE_TARGETS + WEIGHT_COLS   # 6 outputs total

HIDDEN   = 256
DEPTH    = 6
DROPOUT  = 0.10
LR       = 3e-4
EPOCHS   = 600
BATCH    = 512
VAL_FRAC = 0.15
SEED     = 42

W_WEIGHTS = 0.5   # loss weight for species-resolved targets vs gamma/omega

torch.manual_seed(SEED)
np.random.seed(SEED)

# Architecture
class ResBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.LayerNorm(dim),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class TatyanaMLP_V3(nn.Module):
    """
    Single trunk, single head !
    (kymin, trpeps, shat, q0, omt_i, omt_e, omn)
        -> (gamma, omega, gamma_e, omega_e, gamma_i, omega_i)
    kmax / gamma_max resolved analytically via find_peak() at inference time.
    """
    def __init__(self, n_in, n_out, hidden, depth, dropout):
        super().__init__()
        self.embed  = nn.Sequential(nn.Linear(n_in, hidden), nn.SiLU())
        self.blocks = nn.Sequential(*[ResBlock(hidden, dropout) for _ in range(depth)])
        self.head   = nn.Linear(hidden, n_out)

    def forward(self, x):
        return self.head(self.blocks(self.embed(x)))


# Data loading
def load_data(path: Path):
    df = pd.read_csv(path, sep=r"\s+", engine="python")
    df = df[df["is_unstable"] == 1].dropna(subset=FEATURES + ALL_TARGETS)
    X  = df[FEATURES].values.astype(np.float32)
    y  = df[ALL_TARGETS].values.astype(np.float32)
    mode_counts = df["mode"].value_counts().to_dict() if "mode" in df.columns else {}
    print(f"Loaded {len(df)} unstable samples | modes: {mode_counts}")
    return X, y


def make_loaders(X, y, val_frac, batch):
    sx, sy = StandardScaler(), StandardScaler()
    Xs = sx.fit_transform(X).astype(np.float32)
    ys = sy.fit_transform(y).astype(np.float32)
    ds = TensorDataset(torch.from_numpy(Xs), torch.from_numpy(ys))
    n_val = int(len(ds) * val_frac)
    tr, va = random_split(ds, [len(ds) - n_val, n_val],
                          generator=torch.Generator().manual_seed(SEED))
    return (DataLoader(tr, batch_size=batch, shuffle=True),
            DataLoader(va, batch_size=batch),
            sx, sy)


# Training !!
def train(model, tr_loader, va_loader, epochs, lr, device):
    opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    huber  = nn.HuberLoss()
    n_base = len(BASE_TARGETS)   # 2

    history = {"train": [], "val": []}
    best_val, best_state = np.inf, None

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = huber(pred[:, :n_base], yb[:, :n_base]) \
                 + W_WEIGHTS * huber(pred[:, n_base:], yb[:, n_base:])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(tr_loader.dataset)

        model.eval()
        va_loss = 0.
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                va_loss += (huber(pred[:, :n_base], yb[:, :n_base])
                          + W_WEIGHTS * huber(pred[:, n_base:], yb[:, n_base:])
                           ).item() * len(xb)
        va_loss /= len(va_loader.dataset)

        sched.step()
        history["train"].append(tr_loss)
        history["val"].append(va_loss)
        if va_loss < best_val:
            best_val   = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ep % 50 == 0:
            print(f"[{ep:4d}/{epochs}]  train={tr_loss:.5f}  val={va_loss:.5f}  best={best_val:.5f}")

    model.load_state_dict(best_state)
    return history


# Evaluation
def evaluate(model, va_loader, sy, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in va_loader:
            preds.append(model(xb.to(device)).cpu().numpy())
            trues.append(yb.numpy())
    preds = sy.inverse_transform(np.vstack(preds))
    trues = sy.inverse_transform(np.vstack(trues))

    for i, name in enumerate(ALL_TARGETS):
        rel  = np.abs(preds[:, i] - trues[:, i]) / (np.abs(trues[:, i]) + 1e-8)
        rmse = np.sqrt(np.mean((preds[:, i] - trues[:, i])**2))
        print(f"  {name:10s}  RMSE={rmse:.4f}  MedRelErr={np.median(rel)*100:.2f}%")
    return preds, trues


# Plotting
def plot_results(history, preds, trues):
    ncols = 1 + len(ALL_TARGETS)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))

    axes[0].plot(history["train"], label="train")
    axes[0].plot(history["val"],   label="val")
    axes[0].set(xlabel="epoch", ylabel="Huber loss", title="Training curve")
    axes[0].legend(); axes[0].set_yscale("log")

    for i, (ax, name) in enumerate(zip(axes[1:], ALL_TARGETS)):
        ax.scatter(trues[:, i], preds[:, i], alpha=0.3, s=6)
        mn, mx = trues[:, i].min(), trues[:, i].max()
        ax.plot([mn, mx], [mn, mx], "r--", lw=1)
        ax.set(xlabel=f"{name} true", ylabel=f"{name} pred", title=name)

    plt.tight_layout()
    plt.savefig("tatyana_v3_eval.png", dpi=150)
    print("Saved tatyana_v3_eval.png")


# Main
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X, y = load_data(DATA_PATH)
    tr_loader, va_loader, sx, sy = make_loaders(X, y, VAL_FRAC, BATCH)

    model = TatyanaMLP_V3(len(FEATURES), len(ALL_TARGETS), HIDDEN, DEPTH, DROPOUT).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    history = train(model, tr_loader, va_loader, EPOCHS, LR, device)

    print("\nValidation metrics:")
    preds, trues = evaluate(model, va_loader, sy, device)

    torch.save(model.state_dict(), CKPT_PATH)
    joblib.dump({"sx": sx, "sy": sy}, SCALER_PATH)
    print(f"Saved {CKPT_PATH}, {SCALER_PATH}")

    plot_results(history, preds, trues)


# Inference helpers
def load_tatyana_v3(ckpt=CKPT_PATH, scalers=SCALER_PATH, device="cpu"):
    s = joblib.load(scalers)
    model = TatyanaMLP_V3(len(FEATURES), len(ALL_TARGETS), HIDDEN, DEPTH, DROPOUT)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    return model, s["sx"], s["sy"]


def predict(model, sx, sy, X_raw, device="cpu"):
    """
    X_raw : (N, 7)  [kymin, trpeps, shat, q0, omt_i, omt_e, omn]
    Returns (N, 6)  [gamma, omega, gamma_e, omega_e, gamma_i, omega_i]
    """
    Xs = torch.from_numpy(sx.transform(X_raw).astype(np.float32)).to(device)
    with torch.no_grad():
        ys = model(Xs).cpu().numpy()
    return sy.inverse_transform(ys)


def find_peak(model, sx, sy, eq_params, ky_grid=None, device="cpu"):
    """
    Scan gamma(ky) over ky_grid for given equilibrium params and return kmax, gamma_max.

    eq_params : (6,) or (1, 6) — [trpeps, shat, q0, omt_i, omt_e, omn]
    ky_grid   : 1-D array of ky values to scan (default: 50 log-spaced in [0.05, 3.0])
    Returns   : kmax (float), gamma_max (float)
    """
    if ky_grid is None:
        ky_grid = np.logspace(np.log10(0.05), np.log10(3.0), 50).astype(np.float32)

    eq = np.asarray(eq_params, dtype=np.float32).reshape(1, 6)
    eq_rep = np.repeat(eq, len(ky_grid), axis=0)                    # (N_ky, 6)
    X_scan = np.hstack([ky_grid.reshape(-1, 1), eq_rep])            # (N_ky, 7)

    out = predict(model, sx, sy, X_scan, device=device)             # (N_ky, 6)
    gammas = out[:, 0]                                               # gamma column
    idx = int(np.argmax(gammas))
    return float(ky_grid[idx]), float(gammas[idx])


def mode_ratio(model, sx, sy, X_raw, device="cpu"):
    """
    R = |gamma_e / gamma_i| per sample — feeds SAT3 mode-switching M(R).
    X_raw : (N, 7); Returns (N,)
    """
    out = predict(model, sx, sy, X_raw, device=device)
    gamma_e = out[:, 2]   # index 2 in ALL_TARGETS
    gamma_i = out[:, 4]   # index 4 in ALL_TARGETS
    return np.abs(gamma_e / (gamma_i + 1e-8))


if __name__ == "__main__":
    main()
