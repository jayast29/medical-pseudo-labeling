import os
import math
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import models
from sklearn.metrics import roc_auc_score

from datasets import ISICDataset

# ----------------------------
# Config
# ----------------------------
SEED = 42
NUM_CLASSES = 7
EPOCHS = 60
IMG_SIZE = 224

ROOT = "/home/jayastp"
DATASET_ROOT = f"{ROOT}/data/ISIC2018_Task3"
TRAIN_CSV = "data/train_labels.csv"
VAL_CSV   = "data/val_labels.csv"
TEST_CSV  = "data/test_labels.csv"
TRAIN_IMG = f"{DATASET_ROOT}/ISIC2018_Task3_Training_Input"
VAL_IMG   = f"{DATASET_ROOT}/ISIC2018_Task3_Validation_Input"
TEST_IMG  = f"{DATASET_ROOT}/ISIC2018_Task3_Test_Input"

LABELED_FRACTION = 0.20
BATCH_L = 16
RATIO_U = 3
LR      = 1e-4
P_CUTOFF = 0.95
LAMBDA_U = 1.0
LS_ALPHA = 0.10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_datasets():
    full_train = ISICDataset(csv_file=TRAIN_CSV, image_dir=TRAIN_IMG, mode="labeled")
    val_ds     = ISICDataset(csv_file=VAL_CSV,   image_dir=VAL_IMG,   mode="val")
    test_ds    = ISICDataset(csv_file=TEST_CSV,  image_dir=TEST_IMG,  mode="test")

    n_total = len(full_train)
    n_lab   = max(NUM_CLASSES, int(n_total * LABELED_FRACTION))
    labeled_idx   = list(range(n_lab))
    unlabeled_idx = list(range(n_lab, n_total))

    labeled_ds   = Subset(full_train, labeled_idx)
    unlabeled_ds = ISICDataset(csv_file=TRAIN_CSV, image_dir=TRAIN_IMG, mode="unlabeled")
    unlabeled_ds = Subset(unlabeled_ds, unlabeled_idx)

    return labeled_ds, unlabeled_ds, val_ds, test_ds


def make_loaders(labeled_ds, unlabeled_ds, val_ds, test_ds):
    labeled_loader   = DataLoader(labeled_ds,   batch_size=BATCH_L, shuffle=True,  drop_last=True, num_workers=4)
    unlabeled_loader = DataLoader(unlabeled_ds, batch_size=BATCH_L * RATIO_U, shuffle=True, drop_last=True, num_workers=4)
    val_loader       = DataLoader(val_ds,       batch_size=64, shuffle=False, num_workers=4)
    test_loader      = DataLoader(test_ds,      batch_size=64, shuffle=False, num_workers=4)
    return labeled_loader, unlabeled_loader, val_loader, test_loader


def build_model():
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
    return model.to(DEVICE)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total = 0
    correct = 0
    all_probs = []
    all_targets = []

    for x, y in loader:
        x = x.to(DEVICE); y = y.to(DEVICE)
        logits = model(x)
        preds = logits.argmax(dim=1)
        probs = F.softmax(logits, dim=1)

        correct += (preds == y).sum().item()
        total   += y.size(0)

        all_probs.append(probs.cpu())
        all_targets.append(y.cpu())

    acc = 100.0 * correct / max(1, total)
    y_true = torch.cat(all_targets).numpy()
    y_prob = torch.cat(all_probs).numpy()
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except:
        auc = 0.0
    return acc, auc


def train():
    set_seed()
    labeled_ds, unlabeled_ds, val_ds, test_ds = make_datasets()
    lb_loader, ulb_loader, val_loader, test_loader = make_loaders(labeled_ds, unlabeled_ds, val_ds, test_ds)

    model = build_model()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)

    classwise_count = [0 for _ in range(NUM_CLASSES)]
    classwise_acc   = [0.0 for _ in range(NUM_CLASSES)]

    best_val_auc = -1.0
    steps_per_epoch = min(len(ulb_loader), len(lb_loader))

    for epoch in range(1, EPOCHS + 1):
        model.train()
        sup_loss_meter, unsup_loss_meter = 0.0, 0.0

        lb_iter  = iter(lb_loader)
        ulb_iter = iter(ulb_loader)

        for _ in range(steps_per_epoch):
            x_lb, y_lb = next(lb_iter)
            x_w, x_s   = next(ulb_iter)

            x_lb = x_lb.to(DEVICE); y_lb = y_lb.to(DEVICE)
            x_w  = x_w.to(DEVICE);  x_s  = x_s.to(DEVICE)

            logits_lb = model(x_lb)
            sup_loss  = F.cross_entropy(logits_lb, y_lb)

            with torch.no_grad():
                logits_w = model(x_w)
                probs_w  = F.softmax(logits_w, dim=1)
                max_probs, pseudo = probs_w.max(dim=1)

            acc_tensor = torch.tensor(classwise_acc, device=DEVICE)
            tau = P_CUTOFF * (acc_tensor[pseudo] / (2.0 - acc_tensor[pseudo]).clamp_min(1e-6))
            mask = (max_probs >= tau).float()

            logits_s = model(x_s)
            unsup_loss_all = F.cross_entropy(logits_s, pseudo, reduction='none', label_smoothing=LS_ALPHA)
            unsup_loss = (unsup_loss_all * mask).mean()

            loss = sup_loss + LAMBDA_U * unsup_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            sup_loss_meter  += sup_loss.item()
            unsup_loss_meter += unsup_loss.item()

            confident = (max_probs >= P_CUTOFF)
            if confident.any():
                for c in pseudo[confident].tolist():
                    classwise_count[c] += 1
                mx = max(1, max(classwise_count))
                for c in range(NUM_CLASSES):
                    classwise_acc[c] = classwise_count[c] / mx

        scheduler.step()

        val_acc, val_auc = evaluate(model, val_loader)
        test_acc, test_auc = evaluate(model, test_loader)
        avg_sup = sup_loss_meter / steps_per_epoch
        avg_uns = unsup_loss_meter / steps_per_epoch

        print(f"Epoch {epoch:02d} | Sup {avg_sup:.4f}  Unsup {avg_uns:.4f}  "
              f"ValAcc {val_acc:.2f}% AUC {val_auc:.4f} | "
              f"TestAcc {test_acc:.2f}% AUC {test_auc:.4f} | "
              f"C-acc {['%.2f' % a for a in classwise_acc]}")

        # Save best model by val AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), "chkpts/best.pt")

        # Also save every epoch
        torch.save(model.state_dict(), f"chkpts/epoch_{epoch:02d}.pt")


if __name__ == "__main__":
    train()
