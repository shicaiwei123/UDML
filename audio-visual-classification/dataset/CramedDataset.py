import copy
import cv2
import csv
import os
import pickle
import librosa
from scipy import signal
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from torchvision import transforms
import pdb
import time
import numpy as np
import random


def add_gaussian_visual(image, level):
    """Apply QMF Gaussian noise to one PIL image."""
    image = np.asarray(image)
    height, width, channels = image.shape
    noise = np.random.RandomState(0).normal(
        loc=0.0,
        scale=float(level) * 10.0,
        size=(height, width, 1),
    )
    noise = np.repeat(noise, channels, axis=2)
    noisy = noise + image.astype(np.float64)
    noisy[noisy > 255] = 255
    return Image.fromarray(noisy.astype(np.uint8)).convert('RGB')


def add_salt_visual(image, level):
    """Apply 10% salt-and-pepper pixels after probability checks in __getitem__."""
    del level
    image = np.asarray(image).copy()
    height, width, channels = image.shape
    mask = np.random.choice(
        (0, 1, 2),
        size=(height, width, 1),
        p=(0.05, 0.05, 0.90),
    )
    mask = np.repeat(mask, channels, axis=2)
    image[mask == 0] = 0
    image[mask == 1] = 255
    return Image.fromarray(image.astype(np.uint8)).convert('RGB')


def _audio_from_uint8(noisy, source_min, source_range):
    return noisy.astype(np.float32) / 255.0 * source_range + source_min


def _audio_to_uint8(spectrogram):
    source = np.asarray(spectrogram, dtype=np.float32)
    source_min = float(np.min(source))
    source_max = float(np.max(source))
    source_range = source_max - source_min
    if not np.isfinite(source_range) or source_range <= 0.0:
        return source, None
    image = np.clip(
        (source - source_min) / source_range * 255.0,
        0.0,
        255.0,
    ).astype(np.uint8)
    return image, (source_min, source_range)


def add_gaussian_audio(spectrogram, level):
    """Apply QMF Gaussian noise to a raw STFT magnitude."""
    image, scale = _audio_to_uint8(spectrogram)
    if scale is None:
        return image
    noise = np.random.RandomState(0).normal(
        loc=0.0,
        scale=float(level) * 10.0,
        size=image.shape,
    )
    noisy = noise + image.astype(np.float64)
    noisy[noisy > 255] = 255
    return _audio_from_uint8(noisy.astype(np.uint8), *scale)


def add_salt_audio(spectrogram, level):
    """Apply 10% salt-and-pepper pixels to a raw STFT magnitude."""
    del level
    image, scale = _audio_to_uint8(spectrogram)
    if scale is None:
        return image
    mask = np.random.choice((0, 1, 2), size=image.shape, p=(0.05, 0.05, 0.90))
    noisy = image.copy()
    noisy[mask == 0] = 0
    noisy[mask == 1] = 255
    return _audio_from_uint8(noisy, *scale)

class AddMotionBlur(object):
    """
    variance: 控制运动模糊强度（模糊长度）
    """
    def __init__(self, variance=1):
        self.variance = int(variance)

    def __call__(self, img):
        if self.variance <= 1:
            return img

        img = np.array(img)
        h, w, c = img.shape

        # 随机运动方向
        angle = random.uniform(0, 180)

        # 生成 PSF
        ksize = self.variance
        kernel = np.zeros((ksize, ksize))
        kernel[ksize // 2, :] = np.ones(ksize)
        kernel = kernel / ksize

        # 旋转 kernel
        M = cv2.getRotationMatrix2D((ksize / 2, ksize / 2), angle, 1)
        kernel = cv2.warpAffine(kernel, M, (ksize, ksize))

        # 对每个通道做卷积
        blurred = np.zeros_like(img)
        for i in range(c):
            blurred[:, :, i] = cv2.filter2D(img[:, :, i], -1, kernel)

        blurred[blurred > 255] = 255
        blurred[blurred < 0] = 0

        return Image.fromarray(blurred.astype('uint8')).convert('RGB')

from scipy.ndimage import convolve1d

class AddTemporalBlur(object):
    """
    variance: 控制时间模糊强度（卷积核长度）
    """
    def __init__(self, variance=1):
        self.variance = int(variance)

    def __call__(self, spec):
        if self.variance <= 1:
            return spec

        # 时间方向一维卷积
        kernel = np.ones(self.variance) / self.variance
        blurred = convolve1d(spec, kernel, axis=1, mode='nearest')

        return blurred


class AddTimeMask(object):
    """
    variance: 控制时间遮挡比例（0~1）
    """
    def __init__(self, variance=0.1):
        self.variance = variance

    def __call__(self, spec):
        if self.variance <= 0:
            return spec

        F, T = spec.shape
        mask_len = int(T * self.variance)

        t0 = np.random.randint(0, T - mask_len)
        spec[:, t0:t0 + mask_len] = 0

        return spec




class AddOcclusion(object):
    """
    variance: 控制遮挡强度（遮挡块尺寸比例）
    """
    def __init__(self, variance=0.1):
        self.variance = variance

    def __call__(self, img):
        if self.variance <= 0:
            return img

        img = np.array(img)
        h, w,c = img.shape

        # 遮挡块大小
        occ_h = int(h * self.variance)
        occ_w = int(w * self.variance)

        # 随机位置
        top = np.random.randint(0, h - occ_h)
        left = np.random.randint(0, w - occ_w)

        # 遮挡（黑块）
        img[top:top + occ_h, left:left + occ_w, :] = 0

        return Image.fromarray(img.astype('uint8')).convert('RGB')


class AddOcclusion_Aduio(object):
    """
    variance: 控制遮挡强度（遮挡块尺寸比例）
    """
    def __init__(self, variance=0.1):
        self.variance = variance

    def __call__(self, img):
        if self.variance <= 0:
            return img

        img = np.array(img)
        h, w = img.shape

        # 遮挡块大小
        occ_h = int(h * self.variance)
        occ_w = int(w * self.variance)

        # 随机位置
        top = np.random.randint(0, h - occ_h)
        left = np.random.randint(0, w - occ_w)

        # 遮挡（黑块）
        img[top:top + occ_h, left:left + occ_w] = 0

        return img



class CramedDataset(Dataset):

    def __init__(self, args, mode='train',add_noise=False):
        self.args = args
        self.image = []
        self.audio = []
        self.label = []
        self.mode = mode

        self.data_root = './dataset/data/'
        class_dict = {'NEU': 0, 'HAP': 1, 'SAD': 2, 'FEA': 3, 'DIS': 4, 'ANG': 5}

        self.visual_feature_path = args.visual_path
        self.audio_feature_path = args.audio_path

        self.train_csv = os.path.join(self.data_root, args.dataset + '/train.csv')
        self.test_csv = os.path.join(self.data_root, args.dataset + '/test.csv')

        if mode == 'train':
            csv_file = self.train_csv
        else:
            csv_file = self.test_csv

        with open(csv_file, encoding='UTF-8-sig') as f2:
            csv_reader = csv.reader(f2)
            for item in csv_reader:
                audio_path = os.path.join(self.audio_feature_path, item[0] + '.wav')  # wav路径
                visual_path = os.path.join(self.visual_feature_path, 'Image-{:02d}-FPS'.format(self.args.fps),
                                           item[0])  # 包含多个image

                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    self.image.append(visual_path)
                    self.audio.append(audio_path)
                    self.label.append(class_dict[item[1]])
                else:
                    continue

        self.add_noise=add_noise

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        noise_type = getattr(self.args, 'noise_type', 'Gaussian')
        apply_visual_noise = False
        apply_audio_noise = False
        visual_variance = 0
        audio_variance = 0

        if self.add_noise and noise_type != 'None':
            if self.mode == 'train':
                visual_probability = getattr(self.args, 'train_visual_noise_prob', 0.5)
                audio_probability = getattr(self.args, 'train_audio_noise_prob', 0.5)
                visual_candidate = random.randint(
                    getattr(self.args, 'train_visual_variance_min', 0),
                    getattr(self.args, 'train_visual_variance_max', 11),
                )
                audio_candidate = random.randint(
                    getattr(self.args, 'train_audio_variance_min', 0),
                    getattr(self.args, 'train_audio_variance_max', 11),
                )
            else:
                visual_probability = getattr(self.args, 'visual_noise_prob',
                                             getattr(self.args, 'test_visual_noise_prob', 0.5))
                audio_probability = getattr(self.args, 'audio_noise_prob',
                                            getattr(self.args, 'test_audio_noise_prob', 0.5))
                visual_candidate = getattr(self.args, 'visual_variance',
                                           getattr(self.args, 'test_visual_variance', 0))
                audio_candidate = getattr(self.args, 'audio_variance',
                                          getattr(self.args, 'test_audio_variance', 0))

            if visual_probability <= 0:
                apply_visual_noise = False
            elif visual_probability >= 1:
                apply_visual_noise = True
            else:
                apply_visual_noise = random.random() < visual_probability
            if audio_probability <= 0:
                apply_audio_noise = False
            elif audio_probability >= 1:
                apply_audio_noise = True
            else:
                apply_audio_noise = random.random() < audio_probability
            apply_visual_noise = apply_visual_noise and float(visual_candidate) > 0
            apply_audio_noise = apply_audio_noise and float(audio_candidate) > 0

            # QMF Salt uses the level as a second probability gate.
            if noise_type == 'Salt':
                if apply_visual_noise:
                    apply_visual_noise = random.random() < float(visual_candidate) / 100.0
                if apply_audio_noise:
                    apply_audio_noise = random.random() < float(audio_candidate) / 100.0

            if apply_visual_noise:
                visual_variance = np.float32(visual_candidate)
            if apply_audio_noise:
                audio_variance = np.float32(audio_candidate)

        samples, rate = librosa.load(self.audio[idx], sr=22050)
        resamples = np.tile(samples, 3)[:22050 * 3]
        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.

        spectrogram = librosa.stft(resamples, n_fft=512, hop_length=353)
        spectrogram = np.abs(spectrogram)
        if apply_audio_noise:
            if noise_type == 'Gaussian':
                spectrogram = add_gaussian_audio(spectrogram, audio_variance)
            elif noise_type == 'Salt':
                spectrogram = add_salt_audio(spectrogram, audio_variance)
        spectrogram = np.log(np.maximum(spectrogram, 0.0) + 1e-7)
        spectrogram=np.array(spectrogram)

        if self.mode == 'train':
            transform_steps = [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
            ]
        else:
            transform_steps = [transforms.Resize(size=(224, 224))]

        if apply_visual_noise:
            if noise_type == 'Gaussian':
                transform_steps.append(
                    transforms.Lambda(lambda image: add_gaussian_visual(image, visual_variance))
                )
            elif noise_type == 'Salt':
                transform_steps.append(
                    transforms.Lambda(lambda image: add_salt_visual(image, visual_variance))
                )

        transform_steps.extend([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        transform = transforms.Compose(transform_steps)

        # Visual
        image_samples = os.listdir(self.image[idx])

        #control the random
        image_samples.sort()
        seed=int(time.time()*1e6) %2**32
        if len(image_samples)>1:
            select_index = np.random.choice(np.arange(1, len(image_samples)), size=self.args.num_frame, replace=False)
        else:
            select_index=[0]
        select_index.sort()
        
        images = torch.zeros((self.args.num_frame, 3, 224, 224))
        for i in range(self.args.num_frame):
            img = Image.open(os.path.join(self.image[idx], image_samples[select_index[i]])).convert('RGB')
            bt = time.time()
            img = transform(img)
            et = time.time()
            # print(et-bt)
            images[i] = img
        images = torch.permute(images, (1, 0, 2, 3))

        # label
        label = self.label[idx]

        # print(images.shape)

        return spectrogram, images, label,visual_variance,audio_variance

class CramedDataset_swin(Dataset):

    def __init__(self, args, mode='train'):
        self.args = args
        self.image = []
        self.audio = []
        self.label = []
        self.mode = mode

        self.data_root = './dataset/data/'
        class_dict = {'NEU':0, 'HAP':1, 'SAD':2, 'FEA':3, 'DIS':4, 'ANG':5}

        self.visual_feature_path = args.visual_path
        self.audio_feature_path = args.audio_path

        self.train_csv = os.path.join(self.data_root, args.dataset + '/train.csv')
        self.test_csv = os.path.join(self.data_root, args.dataset + '/test.csv')

        if mode == 'train':
            csv_file = self.train_csv
        else:
            csv_file = self.test_csv

        with open(csv_file, encoding='UTF-8-sig') as f2:
            csv_reader = csv.reader(f2)
            for item in csv_reader:
                audio_path = os.path.join(self.audio_feature_path, item[0] + '.wav')  #wav路径
                visual_path = os.path.join(self.visual_feature_path, 'Image-{:02d}-FPS'.format(self.args.fps), item[0])  #包含多个image

                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    self.image.append(visual_path)
                    self.audio.append(audio_path)
                    self.label.append(class_dict[item[1]])
                else:
                    continue


    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):

        # audio
        samples, rate = librosa.load(self.audio[idx], sr=22050)
        resamples = np.tile(samples, 3)[:22050*3]
        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.

        spectrogram = librosa.stft(resamples, n_fft=512, hop_length=353)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)

        spectrogram = np.resize(spectrogram, (224, 224))
        # spectrogram = np.reshape(spectrogram, (1, 224, 224))

        #mean = np.mean(spectrogram)
        #std = np.std(spectrogram)
        #spectrogram = np.divide(spectrogram - mean, std + 1e-9)

        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Visual
        image_samples = os.listdir(self.image[idx])
        select_index = np.random.choice(len(image_samples), size=self.args.fps, replace=False)
        select_index.sort()
        images = torch.zeros((self.args.fps, 3, 224, 224))
        for i in range(self.args.fps):
            img = Image.open(os.path.join(self.image[idx], image_samples[i])).convert('RGB')
            img = transform(img)
            images[i] = img

        images = torch.permute(images, (1,0,2,3))

        # label
        label = self.label[idx]

        return spectrogram, images, label
