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

class AddGaussianNoise(object):

    '''
    mean:均值
    variance：方差
    amplitude：幅值
    '''
    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):

        img = np.array(img)
        h, w, c = img.shape
        # np.random.seed(0)
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance**2, size=(h, w, 1))
        N = np.repeat(N, c, axis=2)
        if self.variance>10:
            img=(N/10)*255.0
        elif self.variance==1:
            img=img
        else:
            img = N + img
        img[img > 255] = 255                       # 避免有值超过255而反转
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img



class AddGaussianNoise_spec(object):

    '''
    mean:均值
    variance：方差
    amplitude：幅值
    '''
    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):

        h, w = img.shape
        np.random.seed(0)
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w))
        if self.variance>10:
            img=(N/10)*255.0
        elif self.variance==1:
            img=img
        else:
            img = N + img
        img[img > 255] = 255                       # 避免有值超过255而反转
        # img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img


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




class AddSaltPepperNoise(object):

    def __init__(self, density=0,p=0.5):
        self.density = density
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:  # 概率的判断
            img = np.array(img)  # 图片转numpy
            h, w, c = img.shape
            Nd = self.density
            Sd = 1 - Nd
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd / 2.0, Nd / 2.0, Sd])  # 生成一个通道的mask
            mask = np.repeat(mask, c, axis=2)  # 在通道的维度复制，生成彩色的mask
            img[mask == 0] = 0  # 椒
            img[mask == 1] = 255  # 盐
            img = Image.fromarray(img.astype('uint8')).convert('RGB')  # numpy转图片
            return img
        else:
            return img

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

        self.a_variance_list=[]
        self.v_variance_list=[]
        for i in range(len(self.image)):
            a = np.float32(np.random.randint(low=1, high=12))
            v=np.float32(np.random.randint(low=1, high=12))
            self.a_variance_list.append(a)
            self.v_variance_list.append(v)

        # 假设self.a_variance_list是一个包含数字的列表
        n = len(self.a_variance_list)
        # 计算需要置为0的元素数量（取一半，向下取整）
        half = n // 2
        # 随机选择一半的索引
        indices_to_zero = random.sample(range(n), half)
        # 将选中的索引位置的元素置为0
        for idx in indices_to_zero:
            self.a_variance_list[idx] = 1
            self.v_variance_list[idx] = 1
        
        self.add_noise=add_noise

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):

        # audio
        # visual_variance=np.float(np.random.randint(low=1, high=12))
        # audio_variance=np.float(np.random.randint(low=1, high=12))
        if self.add_noise:
            if self.mode=='train':
                visual_variance = self.v_variance_list[idx]
                audio_variance = self.a_variance_list[idx]
            else:
                noise_level=0
                p = [0.5, 0.5]
                flag = np.random.choice([0, 1], p=p)
                if flag:
                    visual_variance=noise_level
                else:
                    visual_variance=noise_level
                flag = np.random.choice([0, 1], p=p)
                if flag:
                    audio_variance=noise_level
                else:
                    audio_variance=0
        else:
            visual_variance=1
            audio_variance=1
        
        visual_noise_process=AddGaussianNoise(variance=visual_variance)
        audio_noise_process=AddGaussianNoise_spec(variance=audio_variance)
        samples, rate = librosa.load(self.audio[idx], sr=22050)
        resamples = np.tile(samples, 3)[:22050 * 3]
        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.

        spectrogram = librosa.stft(resamples, n_fft=512, hop_length=353)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)
        if self.mode=='train':
            spectrogram = audio_noise_process(spectrogram)
        else:
            audio_noise_process=AddTemporalBlur(variance=audio_variance)
            spectrogram = audio_noise_process(spectrogram)
        spectrogram=np.array(spectrogram)

        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                visual_noise_process,
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                # AddGaussianNoise(variance=visual_variance),
                transforms.RandomApply([AddMotionBlur(variance=visual_variance)], p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

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