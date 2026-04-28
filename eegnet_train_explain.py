#!/usr/bin/env python3
"""
eegnet_train_explain.py

Trains EEGNet (PyTorch), supports LOSO (leave-one-subject-out).
Produces per-epoch predictions and Integrated Gradients channel saliency.

Usage (example):
python eegnet_train_explain.py --preproc_dir "D:/Data_cog/data_preproc" --out_model "D:/Data_cog/models/eegnet_subjLOSO.pth" --subject_leaveout sub-29 --epochs 40 --batch 64 --gpu 0

Requires: torch, torchvision, captum, numpy, pandas
"""

import os, glob, argparse, time, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from captum.attr import IntegratedGradients

# EEGNet implementation (compact) - adapted for single-channel multi-electrode EEG epochs (channels x time)
class EEGNet(nn.Module):
    def __init__(self, n_chan, n_times, n_classes, Chans=None, Samples=None, dropoutRate=0.5):
        super(EEGNet, self).__init__()
        # This is a minimal EEGNet v1-like architecture
        self.firstConv = nn.Sequential(
            nn.Conv2d(1, 16, (1, 51), padding=(0,25), bias=False),
            nn.BatchNorm2d(16)
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(16, 32, (n_chan,1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1,4)),
            nn.Dropout(p=dropoutRate)
        )
        self.separable = nn.Sequential(
            nn.Conv2d(32, 32, (1,15), padding=(0,7), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1,8)),
            nn.Dropout(p=dropoutRate)
        )
        # compute final dim
        t = n_times
        t = int(np.ceil(t/4/8))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 1 * t, n_classes)
        )

    def forward(self, x):
        # x: (B, 1, channels, times)
        x = self.firstConv(x)
        x = self.depthwise(x)
        x = self.separable(x)
        out = self.classifier(x)
        return out

class EpochDataset(Dataset):
    def __init__(self, files, label_map=None):
        # files: list of tuples (npz_path, task_name)
        rows=[]
        for fn in files:
            data = np.load(fn, allow_pickle=True)
            epochs = data['epochs']  # n_epochs x n_ch x n_times
            task = str(data['task'])
            subj = str(data.get('subject','unknown'))
            for i in range(epochs.shape[0]):
                rows.append((fn, i, task, subj))
        self.rows = rows
        self.label_map = label_map
        if label_map is None:
            le = LabelEncoder()
            le.fit([r[2] for r in rows])
            self.label_map = {c:i for i,c in enumerate(le.classes_)}
        self.label_map = self.label_map

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        fn, i, task, subj = self.rows[idx]
        data = np.load(fn, allow_pickle=True)
        ep = data['epochs'][i].astype(np.float32)  # (n_ch, n_time)
        # ensure shape channels x time
        if ep.shape[0] > ep.shape[1] and data.get('ch_names') is not None and ep.shape[1] == len(data['ch_names']):
            ep = ep.T
        # normalize per epoch (zscore)
        ep = (ep - ep.mean(axis=1, keepdims=True)) / (ep.std(axis=1, keepdims=True) + 1e-8)
        # convert to (1, channels, times)
        ep_tensor = torch.from_numpy(ep[np.newaxis,:,:])
        label = self.label_map[task]
        return ep_tensor, label, fn, i, subj

def collate_fn(batch):
    X = torch.cat([b[0] for b in batch], dim=0)
    y = torch.tensor([b[1] for b in batch], dtype=torch.long)
    meta = [(b[2], b[3], b[4]) for b in batch]
    return X, y, meta

def train_one_epoch(model, loader, opt, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    crit = nn.CrossEntropyLoss()
    for X,y,_ in loader:
        X = X.to(device); y = y.to(device)
        opt.zero_grad()
        out = model(X)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        total_loss += float(loss.item())*X.shape[0]
        preds = out.argmax(1)
        correct += (preds==y).sum().item()
        total += X.shape[0]
    return total_loss/total, correct/total

def eval_model(model, loader, device):
    model.eval()
    total_loss = 0.0; correct = 0; total=0
    preds_all=[]; y_all=[]
    crit = nn.CrossEntropyLoss()
    with torch.no_grad():
        for X,y,_ in loader:
            X = X.to(device); y = y.to(device)
            out = model(X)
            loss = crit(out,y)
            total_loss += float(loss.item())*X.shape[0]
            preds = out.argmax(1)
            preds_all.extend(preds.cpu().numpy().tolist())
            y_all.extend(y.cpu().numpy().tolist())
            correct += (preds==y).sum().item()
            total += X.shape[0]
    return total_loss/total, correct/total, preds_all, y_all

def main(args):
    # collect files
    files = sorted(glob.glob(os.path.join(args.preproc_dir, f"{args.subject_prefix}*.npz")))
    if len(files)==0:
        raise SystemExit("No files")
    print("Found", len(files), "preproc files")
    # For LOSO: leave out subj (if provided) by selecting files accordingly
    # Build dataset
    dataset = EpochDataset(files)
    # get channels/times by loading first epoch
    sample = np.load(files[0], allow_pickle=True)
    ep0 = sample['epochs'][0]
    n_ch, n_times = ep0.shape
    n_classes = len(set([r[2] for r in dataset.rows]))
    print("n_ch, n_times, n_classes:", n_ch, n_times, n_classes)
    # split into train/test by subject_leaveout if provided
    if args.subject_leaveout:
        train_files = [f for f in files if args.subject_leaveout not in f]
        test_files  = [f for f in files if args.subject_leaveout in f]
        if not test_files:
            raise SystemExit("No files for subject_leaveout")
    else:
        train_files = files
        test_files = []

    train_ds = EpochDataset(train_files, label_map=dataset.label_map)
    test_ds  = EpochDataset(test_files, label_map=dataset.label_map) if test_files else None

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=4, collate_fn=collate_fn) if test_ds else None

    device = torch.device(f"cuda:{args.gpu}" if (args.gpu is not None and torch.cuda.is_available()) else "cpu")
    model = EEGNet(n_chan=n_ch, n_times=n_times, n_classes=n_classes).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_acc=0.0
    for epoch in range(1, args.epochs+1):
        t0=time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, opt, device)
        print(f"Epoch {epoch}/{args.epochs} train_loss={train_loss:.4f} train_acc={train_acc:.4f} elapsed={time.time()-t0:.1f}s")
        if test_loader:
            val_loss, val_acc, _, _ = eval_model(model, test_loader, device)
            print(f"   VAL loss={val_loss:.4f} acc={val_acc:.4f}")
            if val_acc>best_acc:
                best_acc=val_acc
                torch.save(model.state_dict(), args.out_model)
                print(" Saved best model.")
    # After training, run IntegratedGradients on some test samples
    if test_loader:
        model.load_state_dict(torch.load(args.out_model))
        model.to(device)
        ig = IntegratedGradients(model)
        # compute IG for first batch of test set
        X,y,meta = next(iter(test_loader))
        X = X.to(device)
        attributions = ig.attribute(X, target=y.to(device), n_steps=50)
        # reduce per-channel (sum abs over time)
        at_np = attributions.detach().cpu().numpy()  # (B,1,channels,times)
        channel_imp = np.mean(np.sum(np.abs(at_np), axis=-1), axis=0).squeeze()  # (channels,)
        # save per-channel importance
        import pandas as pd
        df = pd.DataFrame({'channel_idx': np.arange(len(channel_imp)), 'importance': channel_imp})
        df.to_csv(args.out_model + ".ig_channel_importance.csv", index=False)
        print("Saved IG channel importance to", args.out_model + ".ig_channel_importance.csv")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--preproc_dir', required=True)
    p.add_argument('--subject_prefix', default='sub-')
    p.add_argument('--subject_leaveout', default=None, help="If provided, will use files containing this subject name for test and others for train (LOSO)")
    p.add_argument('--out_model', default='eegnet.pth')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--gpu', type=int, default=0)
    args = p.parse_args()
    main(args)
