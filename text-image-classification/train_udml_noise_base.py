#!/usr/bin/env python3
"""
train_udml.py — UDML training, reusing train_qmf's data pipeline unchanged.

Changes vs train_qmf:
  - Model: MultimodalLateFusionUDML (PE + variance estimator + dynamic fusion)
  - Loss:  cls_loss + beta*KL(mu,std|N(0,1)) + 0.1*MSE(var_pred, 1)
  - Data:  same get_data_loaders, standard 6-tensor batches

Usage:
  python train_udml.py --task MVSA_Single --name udml_clean
"""

import argparse, os, sys, numpy as np, warnings
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from pytorch_pretrained_bert import BertAdam, BertModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from src.data.helpers import get_data_loaders
from src.utils.logger import create_logger
from src.utils.utils import *


# ======================================================================
# UDML Backbone (PE inside encoder)
# ======================================================================

class ImageClf_UDML(nn.Module):
    def __init__(self, fd, n):
        super().__init__()
        m = torchvision.models.resnet152(pretrained=True)
        self.backbone = nn.Sequential(*list(m.children())[:-2])
        self.mu = nn.Sequential(nn.Conv2d(2048, fd, 1, 1, 0), nn.BatchNorm2d(fd))
        self.lv = nn.Sequential(nn.Conv2d(2048, fd, 1, 1, 0), nn.BatchNorm2d(fd))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.clf = nn.Linear(fd, n)
        for s in [self.mu, self.lv]:
            for mm in s.modules():
                if isinstance(mm, nn.Conv2d):
                    nn.init.kaiming_normal_(mm.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(mm, nn.BatchNorm2d):
                    nn.init.constant_(mm.weight, 1); nn.init.constant_(mm.bias, 0)
    def forward(self, img):
        f = self.backbone(img)
        mu = self.mu(f)
        logvar = self.lv(f)
        std = (logvar * 0.5).exp()
        s = mu + torch.randn_like(std) * std if self.training else mu
        return (self.clf(self.pool(s).view(s.size(0), -1)),
                self.pool(s).view(s.size(0), -1),
                self.pool(mu).view(mu.size(0), -1),
                self.pool(std).view(std.size(0), -1))

class BertClf_UDML(nn.Module):
    def __init__(self, fd, n, bert='./bert-base-uncased'):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert)
        self.mu = nn.Sequential(nn.Linear(768, fd), nn.LayerNorm(fd))
        self.lv = nn.Sequential(nn.Linear(768, fd), nn.LayerNorm(fd))
        self.clf = nn.Linear(fd, n)
        for m in [self.mu, self.lv, self.clf]: m.apply(self.bert.init_bert_weights)
    def forward(self, txt, mask, seg):
        _, p = self.bert(txt, token_type_ids=seg, attention_mask=mask, output_all_encoded_layers=False)
        mu = self.mu(p)
        logvar = self.lv(p)
        std = (logvar * 0.5).exp()
        s = mu + torch.randn_like(std) * std if self.training else mu
        return self.clf(s), s, mu, std

def kl_reg(mu, std, tv):
    v = (std ** 2).view(std.size(0), -1)
    mu = mu.view(mu.size(0), -1)
    tv = tv.view(-1, 1)
    return torch.mean(torch.sum(0.5 * (v / tv + mu**2 / tv - torch.log(v / tv + 1e-8) - 1), 1))

class MultimodalLateFusionUDML(nn.Module):
    def __init__(self, args):
        super().__init__(); self.args = args; fd = getattr(args, 'fusion_dim', 512)
        self.txt = BertClf_UDML(fd, args.n_classes, args.bert_model)
        self.img = ImageClf_UDML(fd, args.n_classes)
        self.tve = nn.Sequential(nn.Linear(fd, 256), nn.Dropout(0.1), nn.Linear(256, 1))
        self.ive = nn.Sequential(nn.Linear(fd, 256), nn.Dropout(0.1), nn.Linear(256, 1))
    def forward(self, txt, mask, seg, img):
        tl, ts, tmu, tstd = self.txt(txt, mask, seg)
        il, ims, imu, istd = self.img(img)
        tv = self.tve(tstd.detach())
        iv = self.ive(istd.detach())
        rwt = 2 * iv**2 / (iv**2 + tv**2 + 1e-8)
        rwv = 2 * tv**2 / (iv**2 + tv**2 + 1e-8)
        td = getattr(self.args, 'text_depend', 1.0); vd = getattr(self.args, 'visual_depend', 1.0)
        wt, wv = rwt / td, rwv / vd
        ws = wt + wv + 1e-8; wt, wv = 2 * wt / ws, 2 * wv / ws
        if self.training and hasattr(self.args, 'current_epoch') and \
           self.args.current_epoch < 15:
            wt, wv = torch.ones_like(wt), torch.ones_like(wv)
        # 直接对 logits 加权融合（和 baseline 一样的方式）
        # wt, wv = torch.ones_like(wt)*1.3, torch.ones_like(wv)*0.7

        fo =tl+ wt * tl + il+ wv * il
        fo =wt * tl + wv * il

        return fo, tl, il, tv, iv, wt, wv, tmu, tstd, imu, istd


# ======================================================================
# Training
# ======================================================================

def get_args(parser):
    for a in [("--batch_sz", 32), ("--bert_model", "./bert-base-uncased"),
              ("--data_path", "./datasets"), ("--drop_img_percent", 0.0),
              ("--dropout", 0.1), ("--embed_sz", 300),
              ("--freeze_img", 3), ("--freeze_txt", 5),
              ("--glove_path", ""), ("--gradient_accumulation_steps", 1),
              ("--hidden", []), ("--hidden_sz", 768),
              ("--img_embed_pool_type", "avg"), ("--img_hidden_sz", 2048),
              ("--include_bn", 1), ("--lr", 5e-5), ("--lr_factor", 0.5),
              ("--lr_patience", 2), ("--max_epochs", 100),
              ("--max_seq_len", 512), ("--model", "latefusion_udml"),
              ("--n_workers", 4), ("--name", "udml_run"),
              ("--num_image_embeds", 3), ("--patience", 10),
              ("--savedir", "./checkpoint"), ("--seed", 2),
              ("--task", "MVSA_Single"), ("--task_type", "classification"),
              ("--warmup", 0.1), ("--weight_classes", 1), ("--df", 1),
              ("--noise_level", 0.0), ("--noise_type", "Gaussian")]:
        k, v = a; t = type(v)
        if t == list: parser.add_argument(k, nargs="*", type=int, default=v)
        elif t == bool: parser.add_argument(k, type=int, default=int(v))
        else: parser.add_argument(k, type=t, default=v)
    # UDML
    parser.add_argument("--fusion_dim", type=int, default=1024)
    parser.add_argument("--gamma", type=float, default=4.0)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--cylcle_epoch", type=int, default=10)
    parser.add_argument("--audio_depend", type=float, default=32.0)
    parser.add_argument("--visual_depend", type=float, default=10.0)


@torch.no_grad()
def evaluate(model, loader, criterion, device, verbose=False):
    model.eval()
    losses, preds, tgts = [], [], []
    udml = {'wt':[],'wv':[],'tv':[],'iv':[],'tmu':[],'tstd':[],'imu':[],'istd':[],
            'fp':[],'tp':[],'ip':[],'tgt':[]}
    for batch in loader:
        text, seg, mask, img, target, idx = [x.to(device) for x in batch[:6]]
        fo, tl, il, tvp, ivp, wt, wv, tmu, tstd, imu, istd = model(text, mask, seg, img)
        losses.append(criterion(fo, target).item())
        preds.append(torch.softmax(fo, 1).argmax(1).cpu().numpy())
        tgts.append(target.cpu().numpy())
        if verbose:
            udml['wt'].append(wt.mean().item()); udml['wv'].append(wv.mean().item())
            udml['tv'].append(tvp.mean().item()); udml['iv'].append(ivp.mean().item())
            udml['tmu'].append(tmu.mean().item()); udml['tstd'].append(tstd.mean().item())
            udml['imu'].append(imu.mean().item()); udml['istd'].append(istd.mean().item())
            udml['fp'].append(torch.softmax(fo,1).argmax(1).cpu())
            udml['tp'].append(torch.softmax(tl,1).argmax(1).cpu())
            udml['ip'].append(torch.softmax(il,1).argmax(1).cpu())
            udml['tgt'].append(target.cpu())
    acc = accuracy_score(np.concatenate(tgts), np.concatenate(preds))
    if verbose:
        t = torch.cat(udml['tgt']).numpy()
        s = (f"wt={np.mean(udml['wt']):.4f} wv={np.mean(udml['wv']):.4f} "
             f"tv={np.mean(udml['tv']):.4f} iv={np.mean(udml['iv']):.4f} "
             f"tmu={np.mean(udml['tmu']):.4f} tstd={np.mean(udml['tstd']):.4f} "
             f"imu={np.mean(udml['imu']):.4f} istd={np.mean(udml['istd']):.4f} "
             f"td={getattr(model.args,'text_depend',0):.2f} vd={getattr(model.args,'visual_depend',0):.2f} "
             f"facc={accuracy_score(t,torch.cat(udml['fp']).numpy()):.4f} "
             f"tacc={accuracy_score(t,torch.cat(udml['tp']).numpy()):.4f} "
             f"iacc={accuracy_score(t,torch.cat(udml['ip']).numpy()):.4f}")
    else:
        s = None
    return np.mean(losses), acc, s


def main():
    parser = argparse.ArgumentParser()
    get_args(parser)
    args = parser.parse_args()

    warnings.filterwarnings('ignore')
    torch.backends.cudnn.benchmark = True
    set_seed(args.seed)
    args.savedir = os.path.join(args.savedir, args.name)
    os.makedirs(args.savedir, exist_ok=True)

    train_loader_noise, train_loader_clean, val_loader, test_loaders = get_data_loaders(args, is_udml=True)
    device = torch.device('cuda')

    model = MultimodalLateFusionUDML(args).to(device)
    logger = create_logger(os.path.join(args.savedir, "logfile.log"), args)
    logger.info(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

    best_acc, no_improve = 0, 0

    # ── 重复运行自动走评估（不训练）──
    best_ckpt = f"{args.savedir}/model_best.pt"
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt))
        depend_ckpt = f"{args.savedir}/model_best_depend.pt"
        if os.path.exists(depend_ckpt):
            dep = torch.load(depend_ckpt)
            model.args.text_depend = dep['text_depend']
            model.args.visual_depend = dep['visual_depend']
            logger.info(f"Loaded depend: text={model.args.text_depend:.2f} visual={model.args.visual_depend:.2f}")
        logger.info("Loaded existing model_best.pt, evaluation only:")
        _, _, s = evaluate(model, val_loader, criterion, device, verbose=True)
        if s: logger.info("Val | " + s)
        for name, loader in test_loaders.items():
            l, a, s = evaluate(model, loader, criterion, device, verbose=True)
            msg = f"Test [{name}] loss={l:.4f} acc={a:.4f}"
            if s: msg += " | " + s
            logger.info(msg)
        return

    logger.info("Starting training...")

    for epoch in range(args.max_epochs):
        model.args.current_epoch = epoch

        # 初始 depend
        if not hasattr(model.args, 'text_depend'):
            model.args.text_depend = 1.0
            model.args.visual_depend = 1.0

        # 两阶段噪声切换
        # train_loader.dataset.set_noisy(epoch >= args.cylcle_epoch)
        if epoch<15:
            train_loader=train_loader_clean
        else:
            train_loader=train_loader_noise

        model.train()
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"E{epoch}"):
            text, seg, mask, img, target, idx, tv, iv = [x.to(device) for x in batch]
            fo, tl, il, tvp, ivp, wt, wv, tmu, tstd, imu, istd = model(text, mask, seg, img)
            lf = criterion(fo, target); lt = criterion(tl, target); li = criterion(il, target)
            loss = (lf + args.gamma * (lt + li) +
                    args.beta * (kl_reg(tmu, tstd, tv) + kl_reg(imu, istd, iv)) +
                    0.1 * (F.mse_loss(tvp, tv.view(-1,1)) + F.mse_loss(ivp, iv.view(-1,1))))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_loss += loss.item()
            # 每个 batch 更新 depend
            with torch.no_grad():
                model.args.text_depend = torch.mean(torch.abs(tl), 0).sum().item()
                model.args.visual_depend = torch.mean(torch.abs(il), 0).sum().item()

        print(wt.mean().detach(),wv.mean().detach(),tv.mean().detach(),tvp.mean().detach(),iv.mean().detach(),ivp.mean().detach(),model.args.text_depend,model.args.visual_depend)
        val_loss, val_acc, udml_str = evaluate(model, val_loader, criterion, device, verbose=True)
        scheduler.step(val_acc)
        msg = f"E{epoch} | train_loss={epoch_loss/len(train_loader):.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        if udml_str:
            msg += " | " + udml_str
        logger.info(msg)

        if val_acc > best_acc and epoch>15:
            best_acc = val_acc; no_improve = 0
            torch.save(model.state_dict(), f"{args.savedir}/model_best.pt")
            torch.save({'text_depend': model.args.text_depend,
                        'visual_depend': model.args.visual_depend},
                       f"{args.savedir}/model_best_depend.pt")
        else:
            if epoch>15:
                no_improve += 1
        if no_improve >= args.patience:
            logger.info(f"Early stop at epoch {epoch}, best val_acc={best_acc:.4f}")
            break

    logger.info(f"Done. Best val_acc={best_acc:.4f}")
    model.load_state_dict(torch.load(f"{args.savedir}/model_best.pt"))
    model.eval()
    for name, loader in test_loaders.items():
        l, a, s = evaluate(model, loader, criterion, device, verbose=True)
        msg = f"Test [{name}] loss={l:.4f} acc={a:.4f}"
        if s: msg += " | " + s
        logger.info(msg)

if __name__ == '__main__':
    main()
