#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import functools
import json
import os
import random
from collections import Counter

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from pytorch_pretrained_bert import BertTokenizer
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import JsonlDataset, AddGaussianNoise, AddSaltPepperNoise
from src.data.vocab import Vocab

# ====================================================================
# UDML Dataset（is_udml=True 时启用，不覆盖已有代码）
# ====================================================================


class AddGaussianNoiseUDML:
    """KS/CREMAD 风格高斯噪声"""

    def __init__(self, variance=1.0):
        self.variance = variance

    def __call__(self, img):
        img = np.array(img)
        h, w, c = img.shape
        N = np.random.normal(loc=0, scale=self.variance**2, size=(h, w, 1))
        N = np.repeat(N, c, axis=2)

        if self.variance>10:
            img=(N/10)*255.0
        elif self.variance==1:
            img=img
        else:
            img = N + img

        img[img > 255] = 255
        img[img < 0] = 0
        return Image.fromarray(img.astype('uint8')).convert('RGB')


class UDMLDataset(Dataset):
    def __init__(self, data_path, tokenizer, transform, vocab, args, noisy=False,txt_noise_level=None,img_noise_level=None):
        self.data = [json.loads(l) for l in open(data_path)]
        self.data_dir = os.path.dirname(data_path)
        self.tokenizer = tokenizer
        self.transform = transform
        self.vocab = vocab
        self.args = args
        self.noisy = noisy
        self.max_seq_len = args.max_seq_len
        self.txt_noise_level=txt_noise_level
        self.img_noise_level=img_noise_level

    def set_noisy(self, v):
        self.noisy = v

    def _mask_text(self, text, rate):
        if rate <= 0 or not text:  # 50%
            return text
        words = text.split()
        n = max(1, int(len(words) * rate))
        idx = random.sample(range(len(words)), min(n, len(words)))
        for i in idx:
            words[i] = '_'
        return ' '.join(words)

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        mr = random.random()

        if self.txt_noise_level is not None:
            tv=self.txt_noise_level
        else:
            tv = mr * 10.0 + 1.0
        
        if self.img_noise_level is not None:
            vv=self.img_noise_level
        else:
            vv =  float(np.random.randint(1, 12))
        
        if np.random.random() < 0.5:
            vv = 1.0

        if np.random.random() < 0.5:
            tv = 1.0
            mr=0

        t = item['text']
        if self.noisy:
            t = self._mask_text(t, (tv-1)/10.0)
        else:
            tv = 1
        toks = self.tokenizer(t)
        sent = ["[CLS]"] + toks[:(self.max_seq_len - 1)]
        seg = torch.zeros(len(sent))
        sent = torch.LongTensor(
            [self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"] for w in sent])
        label = torch.LongTensor([self.args.labels.index(item['label'])])
        if item['img']:
            img = Image.open(os.path.join(
                self.data_dir, item['img'])).convert('RGB')
        else:
            img = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))
        if self.noisy and vv > 1:
            img = AddGaussianNoiseUDML(variance=vv)(img)
        else:
            vv = 1
        img = self.transform(img)
        return (sent, seg, img, label, torch.LongTensor([idx]),
                torch.FloatTensor([tv]), torch.FloatTensor([vv]))


def udml_collate(batch):
    lens = [len(r[0]) for r in batch]
    bs, mx = len(batch), max(lens)
    mt = torch.zeros(bs, mx).long()
    tt = torch.zeros(bs, mx).long()
    st = torch.zeros(bs, mx).long()
    it = torch.stack([r[2] for r in batch])
    tg = torch.cat([r[3] for r in batch]).long()
    for i, (r, l) in enumerate(zip(batch, lens)):
        tt[i, :l] = r[0]
        st[i, :l] = r[1]
        mt[i, :l] = 1
    idx = torch.cat([r[4] for r in batch]).long()
    return tt, st, mt, it, tg, idx, torch.cat([r[5] for r in batch]).float(), torch.cat([r[6] for r in batch]).float()


def get_transforms():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.46777044, 0.44531429, 0.40661017],
                std=[0.12221994, 0.12145835, 0.14380469],
            ),
        ]
    )


def get_GaussianNoisetransforms(severity):
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomApply(
                [AddGaussianNoise(variance=severity * 10)], p=0.5),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.46777044, 0.44531429, 0.40661017],
                std=[0.12221994, 0.12145835, 0.14380469],
            ),
        ]
    )


def get_SaltNoisetransforms(severity):
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomApply(
                [AddSaltPepperNoise(density=0.1, p=severity/100)], p=0.5),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.46777044, 0.44531429, 0.40661017],
                std=[0.12221994, 0.12145835, 0.14380469],
            ),
        ]
    )


def get_labels_and_frequencies(path):
    label_freqs = Counter()
    data_labels = [json.loads(line)["label"] for line in open(path)]
    if type(data_labels[0]) == list:
        for label_row in data_labels:
            label_freqs.update(label_row)
    else:
        label_freqs.update(data_labels)

    return list(label_freqs.keys()), label_freqs


def get_glove_words(path):
    word_list = []
    for line in open(path):
        w, _ = line.split(" ", 1)
        word_list.append(w)
    return word_list


def get_vocab(args):
    vocab = Vocab()
    if args.model in ["bert", "mmbt", "concatbert", "latefusion", 'tmc', 'latefusion_udml']:
        bert_tokenizer = BertTokenizer.from_pretrained(
            args.bert_model, do_lower_case=True
        )
        vocab.stoi = bert_tokenizer.vocab
        vocab.itos = bert_tokenizer.ids_to_tokens
        vocab.vocab_sz = len(vocab.itos)

    else:
        word_list = get_glove_words(args.glove_path)
        vocab.add(word_list)

    return vocab


def collate_fn(batch, args):
    lens = [len(row[0]) for row in batch]
    bsz, max_seq_len = len(batch), max(lens)

    mask_tensor = torch.zeros(bsz, max_seq_len).long()
    text_tensor = torch.zeros(bsz, max_seq_len).long()
    segment_tensor = torch.zeros(bsz, max_seq_len).long()

    img_tensor = None
    if args.model in ["img", "concatbow", "concatbert", "mmbt", "latefusion", 'tmc', 'latefusion_udml']:
        img_tensor = torch.stack([row[2] for row in batch])

    if args.task_type == "multilabel":
        tgt_tensor = torch.stack([row[3] for row in batch])
    else:
        tgt_tensor = torch.cat([row[3] for row in batch]).long()

    for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
        tokens, segment = input_row[:2]
        text_tensor[i_batch, :length] = tokens
        segment_tensor[i_batch, :length] = segment
        mask_tensor[i_batch, :length] = 1

    idx = torch.cat([row[4] for row in batch]).long()
    return text_tensor, segment_tensor, mask_tensor, img_tensor, tgt_tensor, idx


def get_data_loaders(args, is_udml=False):
    if is_udml:
        return _get_udml_loaders(args)

    

    tokenizer = (
        BertTokenizer.from_pretrained(
            args.bert_model, do_lower_case=True).tokenize
        if args.model in ["bert", "mmbt", "concatbert", "latefusion", 'tmc', 'latefusion_udml']
        else str.split
    )

    transforms = get_transforms()

    args.labels, args.label_freqs = get_labels_and_frequencies(
        os.path.join(args.data_path, args.task, "train.jsonl")
    )
    vocab = get_vocab(args)
    args.vocab = vocab
    args.vocab_sz = vocab.vocab_sz
    args.n_classes = len(args.labels)

    train = JsonlDataset(
        os.path.join(args.data_path, args.task, "train.jsonl"),
        tokenizer,
        transforms,
        "train",
        vocab,
        args,
    )
    args.train_data_len = len(train)

    dev = JsonlDataset(
        os.path.join(args.data_path, args.task, "dev.jsonl"),
        tokenizer,
        transforms,
        "train",
        vocab,
        args,
    )

    collate = functools.partial(collate_fn, args=args)

    train_loader = DataLoader(
        train,
        batch_size=args.batch_sz,
        shuffle=True,
        num_workers=args.n_workers,
        collate_fn=collate,
    )

    val_loader = DataLoader(
        dev,
        batch_size=args.batch_sz,
        shuffle=False,
        num_workers=args.n_workers,
        collate_fn=collate,
    )

    if args.noise_level > 0.0:
        if args.noise_type == 'Gaussian':
            test_transforms = get_GaussianNoisetransforms(args.noise_level)
        elif args.noise_type == 'Salt':
            test_transforms = get_SaltNoisetransforms(args.noise_level)
    else:
        test_transforms = transforms

    test_set = JsonlDataset(
        os.path.join(args.data_path, args.task, "test.jsonl"),
        tokenizer,
        test_transforms,
        "test",
        vocab,
        args,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_sz,
        shuffle=False,
        num_workers=args.n_workers,
        collate_fn=collate,
    )

    if args.task == "vsnli":
        test_hard = JsonlDataset(
            os.path.join(args.data_path, args.task, "test_hard.jsonl"),
            tokenizer,
            transforms,
            vocab,
            args,
        )

        test_hard_loader = DataLoader(
            test_hard,
            batch_size=args.batch_sz,
            shuffle=False,
            num_workers=args.n_workers,
            collate_fn=collate,
        )

        test = {"test": test_loader, "test_hard": test_hard_loader}
    elif args.task == "MVSA_Single":
        test = {"test": test_loader}

    elif args.task == "food101":
        test = {"test": test_loader}
    else:
        test_gt = JsonlDataset(
            os.path.join(args.data_path, args.task, "test_hard_gt.jsonl"),
            tokenizer,
            test_transforms,
            vocab,
            args,
        )

        test_gt_loader = DataLoader(
            test_gt,
            batch_size=args.batch_sz,
            shuffle=False,
            num_workers=args.n_workers,
            collate_fn=collate,
        )

        test = {
            "test": test_loader,
            "test_gt": test_gt_loader,
        }

    return train_loader, val_loader, test


def _get_udml_loaders(args):
    """UDML 数据加载（is_udml=True 时由 get_data_loaders 调用）"""
    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=True).tokenize
    tfm = get_transforms()

    args.labels, args.label_freqs = get_labels_and_frequencies(
        os.path.join(args.data_path, args.task, "train.jsonl"))
    vocab = get_vocab(args)
    args.vocab = vocab
    args.vocab_sz = vocab.vocab_sz
    args.n_classes = len(args.labels)

    train_noise = UDMLDataset(os.path.join(args.data_path, args.task, 'train.jsonl'),
                              tokenizer, tfm, vocab, args, noisy=True)
    train_clean = UDMLDataset(os.path.join(args.data_path, args.task, 'train.jsonl'),
                              tokenizer, tfm, vocab, args, noisy=False)
    args.train_data_len = len(train_noise)
    dev = UDMLDataset(os.path.join(args.data_path, args.task, 'dev.jsonl'),
                      tokenizer, tfm, vocab, args, noisy=False)

    test = UDMLDataset(os.path.join(args.data_path, args.task, 'test.jsonl'),
                       tokenizer, tfm, vocab, args, noisy=False)

    collate = functools.partial(udml_collate)
    kw = dict(batch_size=args.batch_sz,
              num_workers=args.n_workers, collate_fn=collate)
    train_loader_noise = DataLoader(train_noise, shuffle=True, **kw)
    train_loader_clean = DataLoader(train_clean, shuffle=True, **kw)

    val_loader = DataLoader(dev, shuffle=False, **kw)
    test_loader = DataLoader(test, shuffle=False, **kw)

    if args.task == "vsnli":
        test_hard = UDMLDataset(os.path.join(args.data_path, args.task, 'test_hard.jsonl'),
                                tokenizer, tfm, vocab, args, noisy=False)
        test_hard_loader = DataLoader(test_hard, shuffle=False, **kw)
        test = {"test": test_loader, "test_hard": test_hard_loader}
    elif args.task == "MVSA_Single":
        test = {"test": test_loader}
    elif args.task == "food101":
        test = {"test": test_loader}
    else:
        test_gt = UDMLDataset(os.path.join(args.data_path, args.task, 'test_hard_gt.jsonl'),
                              tokenizer, tfm, vocab, args, noisy=False)
        test_gt_loader = DataLoader(test_gt, shuffle=False, **kw)
        test = {"test": test_loader, "test_gt": test_gt_loader}

    return train_loader_noise, train_loader_clean, val_loader, test


def get_udml_test(args,add_noise,txt_noise_level=None,img_noise_level=None):
    """UDML 数据加载（is_udml=True 时由 get_data_loaders 调用）"""
    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=True).tokenize
    tfm = get_transforms()

    args.labels, args.label_freqs = get_labels_and_frequencies(
        os.path.join(args.data_path, args.task, "train.jsonl"))
    vocab = get_vocab(args)
    args.vocab = vocab
    args.vocab_sz = vocab.vocab_sz
    args.n_classes = len(args.labels)

    test = UDMLDataset(os.path.join(args.data_path, args.task, 'test.jsonl'),
                       tokenizer, tfm, vocab, args, noisy=add_noise,txt_noise_level=txt_noise_level,img_noise_level=img_noise_level)

    collate = functools.partial(udml_collate)
    kw = dict(batch_size=args.batch_sz,
              num_workers=args.n_workers, collate_fn=collate)

    test_loader = DataLoader(test, shuffle=False, **kw)

    if args.task == "vsnli":
        test_hard = UDMLDataset(os.path.join(args.data_path, args.task, 'test_hard.jsonl'),
                                tokenizer, tfm, vocab, args, noisy=add_noise)
        test_hard_loader = DataLoader(test_hard, shuffle=False, **kw)
        test = {"test": test_loader, "test_hard": test_hard_loader}
    elif args.task == "MVSA_Single":
        test = {"test": test_loader}
    elif args.task == "food101":
        test = {"test": test_loader}
    else:
        test_gt = UDMLDataset(os.path.join(args.data_path, args.task, 'test_hard_gt.jsonl'),
                              tokenizer, tfm, vocab, args, noisy=add_noise)
        test_gt_loader = DataLoader(test_gt, shuffle=False, **kw)
        test = {"test": test_loader, "test_gt": test_gt_loader}

    return test