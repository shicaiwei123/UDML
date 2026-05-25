"""
udml.py — UDML standalone training script. Fully self-contained.

Usage:
  python udml.py --model latefusion_udml --task MVSA_Single

No modifications to train_qmf.py or any baseline code.
"""

import argparse, os, sys, json, functools, warnings, random
import numpy as np
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ========================================================================
# 1. Dataset
# ========================================================================

class AddGaussianNoiseRand:
    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):
        self.mean = mean; self.variance = variance; self.amplitude = amplitude
    def __call__(self, img):
        img = np.array(img); h, w, c = img.shape
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
        img = N + np.repeat(N, c, axis=2)
        img[img > 255] = 255; img[img < 0] = 0
        return Image.fromarray(img.astype('uint8')).convert('RGB')


def get_transforms():
    return transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.46777044, 0.44531429, 0.40661017],
                             std=[0.12221994, 0.12145835, 0.14380469]),
    ])


class UDMLDataset(Dataset):
    """Dataset with text masking + variance labels for UDML."""
    def __init__(self, data_path, tokenizer, transform, mode, vocab, args, noisy=False):
        self.data = [json.loads(l) for l in open(data_path)]
        self.data_dir = os.path.dirname(data_path)
        self.tokenizer = tokenizer
        self.transform = transform
        self.vocab = vocab
        self.args = args
        self.mode = mode
        self.noisy = noisy
        self.max_seq_len = args.max_seq_len

    def set_noisy(self, noisy):
        self.noisy = noisy

    def _apply_text_mask(self, text, rate):
        if rate <= 0 or not text:
            return text
        words = text.split()
        n = max(1, int(len(words) * rate))
        idx = random.sample(range(len(words)), min(n, len(words)))
        for i in idx:
            words[i] = '_'
        return ' '.join(words)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Variance targets (always vary)
        mask_rate = random.random()
        text_variance = mask_rate * 10.0 + 1.0
        vis_variance = float(np.random.randint(1, 13))
        if np.random.random() < 0.5:
            vis_variance = 1.0

        # Apply noise only when noisy=True
        raw_text = item['text']
        if self.noisy:
            raw_text = self._apply_text_mask(raw_text, mask_rate)

        # Tokenize
        tokens = self.tokenizer(raw_text)
        sentence = ["[CLS]"] + tokens[:(self.max_seq_len - 1)]
        segment = torch.zeros(len(sentence))
        sentence = torch.LongTensor([
            self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"]
            for w in sentence
        ])

        label = torch.LongTensor([self.args.labels.index(item['label'])])

        # Image
        if item['img']:
            image = Image.open(os.path.join(self.data_dir, item['img'])).convert('RGB')
        else:
            image = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))
        if self.noisy and vis_variance > 1:
            image = AddGaussianNoiseRand(variance=vis_variance)(image)
        image = self.transform(image)

        return (sentence, segment, image, label, torch.LongTensor([idx]),
                torch.FloatTensor([text_variance]), torch.FloatTensor([vis_variance]))


def udml_collate(batch):
    lens = [len(row[0]) for row in batch]
    bsz, max_len = len(batch), max(lens)
    mask_t = torch.zeros(bsz, max_len).long()
    text_t = torch.zeros(bsz, max_len).long()
    seg_t = torch.zeros(bsz, max_len).long()
    img_t = torch.stack([row[2] for row in batch])
    tgt = torch.cat([row[3] for row in batch]).long()
    for i, (row, l) in enumerate(zip(batch, lens)):
        text_t[i, :l] = row[0]; seg_t[i, :l] = row[1]; mask_t[i, :l] = 1
    idx = torch.cat([row[4] for row in batch]).long()
    tv = torch.cat([row[5] for row in batch]).float()
    iv = torch.cat([row[6] for row in batch]).float()
    return text_t, seg_t, mask_t, img_t, tgt, idx, tv, iv


# ========================================================================
# 2. Models
# ========================================================================

class ImageClf_UDML(nn.Module):
    """ResNet152 + Conv2d+BN(mu/logvar) + reparam + pool + FC."""
    def __init__(self, fusion_dim, n_classes):
        super().__init__()
        model = torchvision.models.resnet152(pretrained=True)
        self.backbone = nn.Sequential(*list(model.children())[:-2])
        self.mu_head = nn.Sequential(
            nn.Conv2d(2048, fusion_dim, 1, 1, 0), nn.BatchNorm2d(fusion_dim))
        self.logvar_head = nn.Sequential(
            nn.Conv2d(2048, fusion_dim, 1, 1, 0), nn.BatchNorm2d(fusion_dim))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.clf = nn.Linear(fusion_dim, n_classes)
        for m in [self.mu_head, self.logvar_head]:
            for mm in m.modules():
                if isinstance(mm, nn.Conv2d):
                    nn.init.kaiming_normal_(mm.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(mm, nn.BatchNorm2d):
                    nn.init.constant_(mm.weight, 1); nn.init.constant_(mm.bias, 0)

    def forward(self, img):
        f = self.backbone(img)
        mu = self.mu_head(f); logvar = self.logvar_head(f); std = (logvar * 0.5).exp()
        s = mu + torch.randn_like(std) * std if self.training else mu
        return (self.clf(self.pool(s).view(s.size(0), -1)),
                self.pool(s).view(s.size(0), -1),
                self.pool(mu).view(mu.size(0), -1),
                self.pool(std).view(std.size(0), -1))


class BertClf_UDML(nn.Module):
    """BERT + Linear+LN(mu/logvar) + reparam + FC."""
    def __init__(self, fusion_dim, n_classes, bert_model='./bert-base-uncased'):
        super().__init__()
        from pytorch_pretrained_bert import BertModel
        self.bert = BertModel.from_pretrained(bert_model)
        self.mu_head = nn.Sequential(nn.Linear(768, fusion_dim), nn.LayerNorm(fusion_dim))
        self.logvar_head = nn.Sequential(nn.Linear(768, fusion_dim), nn.LayerNorm(fusion_dim))
        self.clf = nn.Linear(fusion_dim, n_classes)
        for m in [self.mu_head, self.logvar_head, self.clf]:
            m.apply(self.bert.init_bert_weights)

    def forward(self, txt, mask, segment):
        _, pooled = self.bert(txt, token_type_ids=segment, attention_mask=mask, output_all_encoded_layers=False)
        mu = self.mu_head(pooled); logvar = self.logvar_head(pooled); std = (logvar * 0.5).exp()
        s = mu + torch.randn_like(std) * std if self.training else mu
        return self.clf(s), s, mu, std


class ConcatFusion_AUXI(nn.Module):
    def __init__(self, input_dim=1024, output_dim=100):
        super().__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)
    def forward(self, x, y):
        out = self.fc_out(torch.cat((x, y), dim=1))
        xo = self.fc_out(torch.cat((x, torch.zeros_like(y)), dim=1))
        yo = self.fc_out(torch.cat((torch.zeros_like(x), y), dim=1))
        return xo, yo, out


def kl_reguliarize(mu, std, target_var):
    var = std ** 2; var = var.view(var.shape[0], -1)
    mu = mu.view(mu.shape[0], -1)
    tv = target_var.view(-1, 1)
    loss = 0.5 * (var / tv + mu ** 2 / tv - torch.log(var / tv + 1e-8) - 1)
    return torch.mean(torch.sum(loss, dim=1))


class MultimodalLateFusionUDML(nn.Module):
    """UDML: PE encoders -> variance -> depend-scaled weights -> fusion."""
    def __init__(self, args):
        super().__init__()
        self.args = args
        fd = getattr(args, 'fusion_dim', 512)
        self.txtclf = BertClf_UDML(fd, args.n_classes, args.bert_model)
        self.imgclf = ImageClf_UDML(fd, args.n_classes)
        self.txt_ve = nn.Sequential(nn.Linear(fd, 256), nn.Dropout(0.1), nn.Linear(256, 1))
        self.img_ve = nn.Sequential(nn.Linear(fd, 256), nn.Dropout(0.1), nn.Linear(256, 1))
        self.fusion = ConcatFusion_AUXI(fd * 2, args.n_classes)
        self.fusion.apply(self.txtclf.bert.init_bert_weights)

    def forward(self, txt, mask, segment, img):
        tl, ts, tmu, tstd = self.txtclf(txt, mask, segment)
        il, ims, imu, istd = self.imgclf(img)
        tv = (self.txt_ve(tstd.detach()) * 0.5).exp()
        iv = (self.img_ve(istd.detach()) * 0.5).exp()
        rwt, rwv = 2 * iv ** 2 / (iv ** 2 + tv ** 2 + 1e-8), 2 * tv ** 2 / (iv ** 2 + tv ** 2 + 1e-8)
        td, vd = getattr(self.args, 'text_depend', 1.0), getattr(self.args, 'visual_depend', 1.0)
        wt, wv = rwt / td, rwv / vd
        ws_ = wt + wv + 1e-8; wt, wv = 2 * wt / ws_, 2 * wv / ws_
        if self.training and hasattr(self.args, 'current_epoch') and \
           self.args.current_epoch < self.args.cylcle_epoch + 10:
            wt, wv = torch.ones_like(wt), torch.ones_like(wv)
        fx, fy, fo = self.fusion(ts + wt * ts, ims + wv * ims)
        return fo, tl, il, tv, iv, wt, wv, tmu, tstd, imu, istd, fx, fy


# ========================================================================
# 3. Training
# ========================================================================

def train_epoch(model, loader, optimizer, criterion, args, epoch):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc=f"Epoch {epoch}"):
        text, seg, mask, img, target, idx, tv, iv = [x.cuda() for x in batch]
        fo, tl, il, tvp, ivp, wt, wv, tmu, tstd, imu, istd, fx, fy = model(text, mask, seg, img)
        lf = criterion(fo, target); lt = criterion(tl, target); li = criterion(il, target)
        lfx = criterion(fx, target); lfy = criterion(fy, target)
        lcls = lf + args.gamma * (lt + li + lfx + lfy)
        lkl = kl_reguliarize(tmu, tstd, tv) + kl_reguliarize(imu, istd, iv)
        lvar = F.mse_loss(tvp, tv.view(-1, 1)) + F.mse_loss(ivp, iv.view(-1, 1))
        loss = lcls + args.beta * lkl + lvar * 0.1
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, all_pred, all_tgt = 0, [], []
    for batch in loader:
        text, seg, mask, img, target, idx, tv, iv = [x.cuda() for x in batch]
        fo, tl, il, tvp, ivp, wt, wv, tmu, tstd, imu, istd, fx, fy = model(text, mask, seg, img)
        lf = criterion(fo, target)
        total_loss += lf.item()
        pred = torch.softmax(fo, dim=1).argmax(dim=1)
        all_pred.append(pred.cpu().numpy()); all_tgt.append(target.cpu().numpy())
    return total_loss / len(loader), accuracy_score(np.concatenate(all_tgt), np.concatenate(all_pred))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='latefusion_udml')
    parser.add_argument('--task', default='MVSA_Single')
    parser.add_argument('--data_path', default='./datasets')
    parser.add_argument('--batch_sz', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--name', default='udml_run')
    parser.add_argument('--savedir', default='./checkpoint')
    parser.add_argument('--bert_model', default='./bert-base-uncased')
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--fusion_dim', type=int, default=512)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--cylcle_epoch', type=int, default=10)
    parser.add_argument('--noise_free', type=int, default=1)
    args = parser.parse_args()

    warnings.filterwarnings('ignore')
    torch.backends.cudnn.benchmark = True

    # ── Data ──
    from pytorch_pretrained_bert import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True).tokenize
    with open(os.path.join(args.data_path, args.task, 'train.jsonl')) as f:
        labels = list(Counter([json.loads(l)['label'] for l in f]).keys())
    args.labels = labels; args.n_classes = len(labels)
    print(f"Labels: {labels} ({args.n_classes} classes)")

    bt = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
    class Vocab: pass
    vocab = Vocab(); vocab.stoi = bt.vocab; vocab.itos = bt.ids_to_tokens; vocab.vocab_sz = len(vocab.itos)

    tfm = get_transforms()
    train_set = UDMLDataset(os.path.join(args.data_path, args.task, 'train.jsonl'),
                            tokenizer, tfm, 'train', vocab, args, noisy=False)
    dev_set = UDMLDataset(os.path.join(args.data_path, args.task, 'dev.jsonl'),
                          tokenizer, tfm, 'dev', vocab, args, noisy=False)
    train_loader = DataLoader(train_set, args.batch_sz, shuffle=True, num_workers=args.n_workers, collate_fn=udml_collate)
    dev_loader = DataLoader(dev_set, args.batch_sz, shuffle=False, num_workers=args.n_workers, collate_fn=udml_collate)
    print(f"Train: {len(train_set)}, Dev: {len(dev_set)}")

    # ── Model ──
    model = MultimodalLateFusionUDML(args).cuda()
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    # ── Train ──
    best_acc, no_improve = 0, 0
    os.makedirs(f"{args.savedir}/{args.name}", exist_ok=True)

    for epoch in range(args.max_epochs):
        model.args.current_epoch = epoch
        if args.noise_free:
            train_set.set_noisy(False)
        else:
            train_set.set_noisy(epoch >= args.cylcle_epoch)

        # Compute depend once at epoch 0
        if epoch == 0:
            model.eval()
            with torch.no_grad():
                batch = next(iter(train_loader))
                text, seg, mask, img = [x.cuda() for x in batch[:4]]
                _, tl, il, *_ = model(text, mask, seg, img)
                model.args.text_depend = torch.mean(torch.abs(tl), 0).sum().item()
                model.args.visual_depend = torch.mean(torch.abs(il), 0).sum().item()
            print(f"  text_depend={model.args.text_depend:.2f}, visual_depend={model.args.visual_depend:.2f}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, args, epoch)
        val_loss, val_acc = evaluate(model, dev_loader, criterion)
        scheduler.step(val_acc)
        print(f"  Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc; no_improve = 0
            torch.save(model.state_dict(), f"{args.savedir}/{args.name}/model_best.pt")
        else:
            no_improve += 1
        if no_improve >= args.patience:
            print(f"Early stop at epoch {epoch}, best val_acc={best_acc:.4f}")
            break

    print(f"\nDone. Best val_acc: {best_acc:.4f}")

if __name__ == '__main__':
    main()
