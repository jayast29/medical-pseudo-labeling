"""
FaxMatch Training - Fixed with True MCPL (Multi-Curriculum Pseudo-Labeling)
Implements Equations 1-5 from the paper correctly
"""

import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.models import densenet121, DenseNet121_Weights
import pandas as pd
from PIL import Image
import torchvision.transforms as T
from sklearn.metrics import roc_auc_score


# ==================== Dataset ====================
class ISICDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, image_dir, mode="labeled"):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.mode = mode
        
        self.weak_transform = T.Compose([
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.strong_transform = T.Compose([
            T.Resize((224, 224)),
            T.RandAugment(num_ops=2, magnitude=9),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.eval_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]["image_id"]
        img_path = os.path.join(self.image_dir, img_id + ".jpg")
        image = Image.open(img_path).convert("RGB")
        
        if self.mode == "labeled" or self.mode == "val" or self.mode == "test":
            label = self.df.iloc[idx]["label"]
            if self.mode == "labeled":
                image = self.weak_transform(image)
            else:
                image = self.eval_transform(image)
            return image, label
        
        elif self.mode == "unlabeled":
            weak = self.weak_transform(image)
            strong = self.strong_transform(image)
            return weak, strong


# ==================== True MCPL Threshold Update ====================
@torch.no_grad()
def update_mcpl_thresholds(model, labeled_loader, unlabeled_loader, 
                           T=0.95, beta=0.6, num_classes=7, device='cuda'):
    """
    Multi-Curriculum Pseudo-Labeling (Equations 1-5 from paper)
    
    Args:
        model: Neural network
        labeled_loader: DataLoader for labeled data
        unlabeled_loader: DataLoader for unlabeled data (returns weak, strong pairs)
        T: Initial fixed threshold (default 0.95)
        beta: Weight for labeled vs unlabeled learning effect
        num_classes: Number of classes
        device: cuda or cpu
    
    Returns:
        torch.Tensor: Dynamic thresholds per class [num_classes]
    """
    model.eval()
    
    # Track labeled data statistics
    class_dist = torch.zeros(num_classes)
    labeled_correct = torch.zeros(num_classes)
    
    # Evaluate on labeled data (Equation 2)
    for x, y in labeled_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        
        for c in range(num_classes):
            mask = y == c
            class_dist[c] += mask.sum().item()
            labeled_correct[c] += ((preds == c) & mask).sum().item()
    
    # Track unlabeled data predictions (Equation 3)
    unlabeled_preds = torch.zeros(num_classes)
    total_unlabeled = 0
    
    for x_w, x_s in unlabeled_loader:
        x_w = x_w.to(device)
        logits = model(x_w)
        preds = logits.argmax(dim=1)
        
        for c in range(num_classes):
            unlabeled_preds[c] += (preds == c).sum().item()
        total_unlabeled += len(x_w)
    
    # Calculate thresholds per class
    thresholds = []
    total_labeled = class_dist.sum()
    
    for c in range(num_classes):
        if class_dist[c] == 0:
            # No labeled samples for this class
            thresholds.append(T)
            continue
        
        # Equation 1: Estimate unlabeled class distribution
        S_c_u = (class_dist[c] / total_labeled) * total_unlabeled
        
        # Equation 2: Learning effect on labeled data (accuracy)
        delta_e_c = labeled_correct[c] / class_dist[c]
        
        # Equation 3: Learning effect on unlabeled data (prediction ratio)
        epsilon_e_c = unlabeled_preds[c] / S_c_u if S_c_u > 0 else 0
        
        # Equation 4: Combined learning status
        sigma_e_c = beta * delta_e_c + (1 - beta) * epsilon_e_c
        
        # Equation 5: Adjust threshold
        t_e_c = T * sigma_e_c
        
        # Avoid extreme cases (from paper)
        if epsilon_e_c > 1.0 or t_e_c > 1.0:
            t_e_c = T
        
        # Prevent collapse (paper uses minimum ~0.3 for minority classes)
        t_e_c = max(t_e_c, 0.30)
        
        thresholds.append(t_e_c)
    
    model.train()
    return torch.tensor(thresholds, device=device)


# ==================== Evaluation ====================
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    all_probs = []
    all_targets = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        probs = F.softmax(logits, dim=1)

        correct += (preds == y).sum().item()
        total += y.size(0)

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


# ==================== Training ====================
def train_faxmatch():
    # Config
    cfg = {
        'seed': 42,
        'train_csv': 'data/train_labels.csv',
        'val_csv': 'data/val_labels.csv',
        'test_csv': 'data/test_labels.csv',
        'img_dir_train': '/home/jayastp/data/ISIC2018_Task3/ISIC2018_Task3_Training_Input',
        'img_dir_val': '/home/jayastp/data/ISIC2018_Task3/ISIC2018_Task3_Validation_Input',
        'img_dir_test': '/home/jayastp/data/ISIC2018_Task3/ISIC2018_Task3_Test_Input',
        'num_classes': 7,
        'labeled_fraction': 0.2,
        'batch_l': 16,
        'ratio_u': 3,
        'epochs': 60,
        'lr': 0.003,
        'lambda_u': 1.0,
        'ls_alpha': 0.1,
        'beta_mcpl': 0.6,  # MCPL weight for labeled vs unlabeled
        'p_cutoff': 0.95,
        'mcpl_update_freq': 5,  # Update thresholds every N epochs
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("="*70)
    print("FaxMatch Training with True MCPL")
    print("="*70)
    print(f"Device: {cfg['device']}")
    print(f"Labeled: {cfg['labeled_fraction']*100}%")
    print(f"MCPL update frequency: Every {cfg['mcpl_update_freq']} epochs")
    print("="*70 + "\n")
    
    # Set seed
    import random
    random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    
    # Datasets
    full_train = ISICDataset(csv_file=cfg['train_csv'], image_dir=cfg['img_dir_train'], mode="labeled")
    val_ds = ISICDataset(csv_file=cfg['val_csv'], image_dir=cfg['img_dir_val'], mode="val")
    test_ds = ISICDataset(csv_file=cfg['test_csv'], image_dir=cfg['img_dir_test'], mode="test")
    
    n_total = len(full_train)
    n_lab = max(cfg['num_classes'], int(n_total * cfg['labeled_fraction']))
    labeled_idx = list(range(n_lab))
    unlabeled_idx = list(range(n_lab, n_total))
    
    labeled_ds = Subset(full_train, labeled_idx)
    unlabeled_ds = ISICDataset(csv_file=cfg['train_csv'], image_dir=cfg['img_dir_train'], mode="unlabeled")
    unlabeled_ds = Subset(unlabeled_ds, unlabeled_idx)
    
    print(f"Dataset split: Labeled={len(labeled_ds)}, Unlabeled={len(unlabeled_ds)}\n")
    
    # Loaders
    batch_u = cfg['batch_l'] * cfg['ratio_u']
    lb_loader = DataLoader(labeled_ds, batch_size=cfg['batch_l'], shuffle=True, 
                          drop_last=True, num_workers=4, pin_memory=True)
    ulb_loader = DataLoader(unlabeled_ds, batch_size=batch_u, shuffle=True, 
                           drop_last=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    
    # Model
    model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, cfg['num_classes'])
    model = model.to(cfg['device'])
    
    # Optimizer & Scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg['lr'], 
                               momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
    
    # Initialize thresholds (all start at 0.95)
    threshold_per_class = torch.full((cfg['num_classes'],), cfg['p_cutoff']).to(cfg['device'])
    
    # Logging
    os.makedirs('threshold_logs', exist_ok=True)
    os.makedirs('chkpts', exist_ok=True)
    log_file = open('threshold_logs/thresholds_log.csv', 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['epoch'] + [f'class_{i}' for i in range(cfg['num_classes'])])
    
    best_val_auc = -1.0
    steps_per_epoch = min(len(lb_loader), len(ulb_loader))
    
    # Training loop
    for epoch in range(1, cfg['epochs'] + 1):
        # Update thresholds using MCPL (every N epochs)
        if epoch % cfg['mcpl_update_freq'] == 0 or epoch == 1:
            print(f"[Epoch {epoch}] Updating MCPL thresholds...", flush=True)
            threshold_per_class = update_mcpl_thresholds(
                model=model,
                labeled_loader=lb_loader,
                unlabeled_loader=ulb_loader,
                T=cfg['p_cutoff'],
                beta=cfg['beta_mcpl'],
                num_classes=cfg['num_classes'],
                device=cfg['device']
            )
        
        model.train()
        sup_loss_meter = unsup_loss_meter = 0.0
        
        lb_iter = iter(lb_loader)
        ulb_iter = iter(ulb_loader)
        
        for _ in range(steps_per_epoch):
            # Get batches
            x_lb, y_lb = next(lb_iter)
            x_w, x_s = next(ulb_iter)
            
            x_lb, y_lb = x_lb.to(cfg['device']), y_lb.to(cfg['device'])
            x_w, x_s = x_w.to(cfg['device']), x_s.to(cfg['device'])
            
            # Supervised loss
            logits_lb = model(x_lb)
            sup_loss = F.cross_entropy(logits_lb, y_lb)
            
            # Pseudo-labels from weak augmentation
            with torch.no_grad():
                logits_w = model(x_w)
                probs_w = F.softmax(logits_w, dim=1)
                max_probs, pseudo = probs_w.max(dim=1)
                
                # Class-wise dynamic thresholding (key FaxMatch innovation)
                mask = max_probs.ge(threshold_per_class[pseudo]).float()
            
            # Unsupervised loss with label smoothing
            logits_s = model(x_s)
            pseudo_ls = F.one_hot(pseudo, cfg['num_classes']).float()
            pseudo_ls = (1 - cfg['ls_alpha']) * pseudo_ls + cfg['ls_alpha'] / cfg['num_classes']
            unsup_loss_all = F.cross_entropy(logits_s, pseudo_ls, reduction='none')
            unsup_loss = (unsup_loss_all * mask).mean()
            
            # Total loss
            loss = sup_loss + cfg['lambda_u'] * unsup_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            sup_loss_meter += sup_loss.item()
            unsup_loss_meter += unsup_loss.item()
        
        scheduler.step()
        
        # Evaluate
        val_acc, val_auc = evaluate(model, val_loader, cfg['device'])
        test_acc, test_auc = evaluate(model, test_loader, cfg['device'])
        
        avg_sup = sup_loss_meter / steps_per_epoch
        avg_unsup = unsup_loss_meter / steps_per_epoch
        
        # Log thresholds
        threshold_list = threshold_per_class.cpu().tolist()
        log_writer.writerow([epoch] + threshold_list)
        log_file.flush()
        
        # Print results
        print(f"Epoch {epoch:02d} | Sup {avg_sup:.4f}  Unsup {avg_unsup:.4f}  "
              f"ValAcc {val_acc:.2f}% AUC {val_auc:.4f} | "
              f"TestAcc {test_acc:.2f}% AUC {test_auc:.4f}")
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_auc': val_auc,
                'test_auc': test_auc,
                'thresholds': threshold_per_class
            }, "chkpts/best.pt")
        
        # Save periodic checkpoints
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"chkpts/epoch_{epoch:02d}.pt")
    
    log_file.close()
    
    print("\n" + "="*70)
    print(f"Training Complete! Best Val AUC: {best_val_auc:.4f}")
    print("="*70)


if __name__ == '__main__':
    train_faxmatch()