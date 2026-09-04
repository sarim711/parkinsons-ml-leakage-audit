"""
Neural models (PyTorch). Optional — requires `torch`.

Usage:
    python neural.py mlp3       # 3-class MLP            (39 -> 128 -> 64 -> 3)
    python neural.py mlpdeep    # 3-class deeper MLP     (39 -> 256 -> 128 -> 64 -> 3)
    python neural.py attn       # 3-class FeatureAttention (FT-Transformer-lite)
    python neural.py mlp2       # 2-class MLP            (39 -> 128 -> 64 -> 2)

Training: AdamW + cosine LR schedule + gradient clipping, class-weighted cross-entropy,
features standardized per fold. Results printed as mean ± sample std across folds.

Importing this module has no global side effects: the warning filter and the torch
thread count are applied inside main(), i.e. only when it is run as a script.
"""
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score

import preprocessing as P

USAGE = "usage: python neural.py {mlp3|mlpdeep|attn|mlp2}"

# Thread count used to produce the reported numbers. Applied in main() only.
TORCH_NUM_THREADS = 4


def seed_all(s=P.SEED):
    np.random.seed(s); torch.manual_seed(s)


class MLP(nn.Module):
    """Linear -> BatchNorm -> GELU -> Dropout blocks, then a linear head."""
    def __init__(self, d_in, hidden, n_out, p):
        super().__init__()
        layers, prev = [], d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(p)]
            prev = h
        layers += [nn.Linear(prev, n_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FeatureAttention(nn.Module):
    """FT-Transformer-lite: each scalar feature -> token, CLS token, transformer encoder."""
    def __init__(self, n_feat, n_out, d=16, heads=4, layers=2):
        super().__init__()
        self.value = nn.Parameter(torch.randn(n_feat, d) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_feat, d))
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=4 * d,
                                         dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Linear(d, n_out)

    def forward(self, x):
        b = x.shape[0]
        tokens = x.unsqueeze(-1) * self.value.unsqueeze(0) + self.bias.unsqueeze(0)
        seq = torch.cat([self.cls.expand(b, -1, -1), tokens], dim=1)
        return self.head(self.encoder(seq)[:, 0])


def train_eval(kind, y, n_out, folds, Xv, hidden, dropout, epochs, lr=2e-3):
    acc, f1, bacc, auc = [], [], [], []
    for tr, te in folds:
        seed_all()
        sc = StandardScaler()
        a = torch.tensor(sc.fit_transform(Xv[tr]), dtype=torch.float32)
        b = torch.tensor(sc.transform(Xv[te]), dtype=torch.float32)
        yt = torch.tensor(y[tr], dtype=torch.long)
        cw = torch.tensor(compute_class_weight("balanced", classes=np.arange(n_out), y=y[tr]),
                          dtype=torch.float32)
        model = (FeatureAttention(Xv.shape[1], n_out) if kind == "attn"
                 else MLP(Xv.shape[1], hidden, n_out, dropout))
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        loss_fn = nn.CrossEntropyLoss(weight=cw)
        n, bs = len(a), 512
        for _ in range(epochs):
            model.train()
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                loss = loss_fn(model(a[idx]), yt[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            prob = torch.softmax(model(b), 1).numpy()
        pred = prob.argmax(1)
        yte = y[te]
        acc.append(accuracy_score(yte, pred))
        f1.append(f1_score(yte, pred, average="macro"))
        bacc.append(balanced_accuracy_score(yte, pred))
        if n_out == 2:
            auc.append(roc_auc_score(yte, prob[:, 1]) if len(np.unique(yte)) > 1 else 0.5)
    sd = lambda x: np.std(x, ddof=1)
    out = (f"acc={np.mean(acc):.3f}±{sd(acc):.3f} f1={np.mean(f1):.3f}±{sd(f1):.3f} "
           f"bacc={np.mean(bacc):.3f}±{sd(bacc):.3f}")
    if n_out == 2:
        out += f" auc={np.mean(auc):.3f}±{sd(auc):.3f}"
    return out


def main():
    if len(sys.argv) < 2:
        raise SystemExit(USAGE)
    which = sys.argv[1]
    if which not in {"mlp3", "mlpdeep", "attn", "mlp2"}:
        raise SystemExit(USAGE)

    # Script-only side effects, kept here so `import neural` stays clean.
    warnings.filterwarnings("ignore")
    torch.set_num_threads(TORCH_NUM_THREADS)

    df = P.load_raw()
    X, y_reg, y3, y2, groups = P.get_xy(df)
    Xv = X.values.astype(np.float32)
    if which == "mlp3":
        f = P.make_folds(df, y3, groups)
        print("MLP 3-class      ", train_eval("mlp", y3, 3, f, Xv, [128, 64], 0.3, 40))
    elif which == "mlpdeep":
        f = P.make_folds(df, y3, groups)
        print("MLP_deeper 3-class", train_eval("mlp", y3, 3, f, Xv, [256, 128, 64], 0.4, 50))
    elif which == "attn":
        f = P.make_folds(df, y3, groups)
        print("FeatureAttention ", train_eval("attn", y3, 3, f, Xv, None, None, 25))
    elif which == "mlp2":
        f = P.make_folds(df, y2, groups)
        print("MLP 2-class      ", train_eval("mlp", y2, 2, f, Xv, [128, 64], 0.3, 40))


if __name__ == "__main__":
    main()
