import copy
import csv
import os
import pickle
import librosa
from scipy import signal
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import skimage
import random
import time
from PIL import Image, ImageFilter
import pdb
import torch.nn as nn
import glob
import numpy as np
import time


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

def listdir_nohidden(path):
    data_list= glob.glob(os.path.join(path, '*'))
    data_list.sort()
    return data_list


class KSDataset_Noise(nn.Module):
    def __init__(self, args,add_noise=False, mode='train', data_path='./train_test_data/kinect_sound'):
        super().__init__()

        f = open('dataset/data/KineticSound/class.txt')
        data = f.readline()
        class_list = data.split(',')
        for i in range(len(class_list)):
            if " " in class_list[i]:
                class_name = class_list[i].split(" ")
                if class_name[0] == '':
                    class_name = class_name[1:len(class_name)]
                class_name = '_'.join(class_name)
                class_list[i] = class_name

        self.args = args

        label = range(len(class_list))
        data_dict = zip(class_list, label)
        data_dict = dict(data_dict)

        self.mode = mode
        if self.mode == 'train':
            visual_data_path = os.path.join(data_path, 'visual', 'train_img/Image-01-FPS')
            audio_data_path = os.path.join(data_path, 'audio', 'train')
        elif self.mode == 'test':
            visual_data_path = os.path.join(data_path, 'visual', 'val_img/Image-01-FPS')
            audio_data_path = os.path.join(data_path, 'audio', 'test')

        self.data_label = []
        self.video_path_list = []
        self.audio_path_list = []

        remove_list = []  # 移除损坏视频

        # i=0
        for class_name in class_list:
            visual_class_path = os.path.join(visual_data_path, class_name)
            audio_class_path = os.path.join(audio_data_path, class_name)

            video_list = os.listdir(visual_class_path)
            video_list.sort()

            audio_list = os.listdir(audio_class_path)
            audio_list.sort()

            for video in video_list:
                # i+=1
                video_path = os.path.join(visual_class_path, video)

                if len(listdir_nohidden(video_path)) < 3:
                    # print(video_path)
                    remove_list.append(video)
                    continue

                self.video_path_list.append(video_path)
                self.data_label.append(data_dict[class_name])

            for audio in audio_list:
                if audio in remove_list:
                    print(audio)
                    continue
                audio_path = os.path.join(audio_class_path, audio)
                self.audio_path_list.append(audio_path)

        self.add_noise=add_noise


    def __len__(self):
        # return 10000

        return len(self.data_label)

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
        
        # audio
        sample, rate = librosa.load(self.audio_path_list[idx], sr=16000, mono=True)
        while len(sample) / rate < 10.:
            sample = np.tile(sample, 2)

        start_point = random.randint(a=0, b=rate * 5)
        new_sample = sample[start_point:start_point + rate * 5]
        new_sample[new_sample > 1.] = 1.
        new_sample[new_sample < -1.] = -1.

        spectrogram = librosa.stft(new_sample, n_fft=256, hop_length=128)
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
        image_samples = listdir_nohidden(self.video_path_list[idx])
        # print(len(image_samples))
        select_index = np.random.choice(len(image_samples), size=self.args.num_frame, replace=False)
        select_index.sort()
        images = torch.zeros((self.args.num_frame, 3, 224, 224))
        for i in range(self.args.num_frame):
            try:
                img = Image.open(image_samples[select_index[i]]).convert('RGB')
            except Exception as e:
                print(e)
                print(image_samples[i])
                continue

            bt = time.time()
            img = transform(img)
            et = time.time()
            # print(et-bt)
            images[i] = img

        images = torch.permute(images, (1, 0, 2, 3))

        # label
        label = self.data_label[idx]
        # print(label)

        return spectrogram, images, label,visual_variance,audio_variance



class KSDataset_swin(nn.Module):
    def __init__(self, args, mode='train', data_path='./train_test_data/kinect_sound'):
        super().__init__()

        f = open('dataset/data/KineticSound/class.txt')
        data = f.readline()
        class_list = data.split(',')
        for i in range(len(class_list)):
            if " " in class_list[i]:
                class_name = class_list[i].split(" ")
                if class_name[0] == '':
                    class_name = class_name[1:len(class_name)]
                class_name = '_'.join(class_name)
                class_list[i] = class_name

        self.args = args

        # class_list=[class_list[0],class_list[1]]

        label = range(len(class_list))
        data_dict = zip(class_list, label)
        data_dict = dict(data_dict)

        # print(data_dict)

        self.mode = mode
        if self.mode == 'train':
            visual_data_path = os.path.join(data_path, 'visual', 'train_img/Image-01-FPS')
            audio_data_path = os.path.join(data_path, 'audio', 'train')
        elif self.mode == 'test':
            visual_data_path = os.path.join(data_path, 'visual', 'val_img/Image-01-FPS')
            audio_data_path = os.path.join(data_path, 'audio', 'test')

        self.data_label = []
        self.video_path_list = []
        self.audio_path_list = []

        remove_list = []  # 移除损坏视频

        # i=0
        for class_name in class_list:
            visual_class_path = os.path.join(visual_data_path, class_name)
            audio_class_path = os.path.join(audio_data_path, class_name)

            video_list = os.listdir(visual_class_path)
            video_list.sort()

            audio_list = os.listdir(audio_class_path)
            audio_list.sort()

            for video in video_list:
                # i+=1
                video_path = os.path.join(visual_class_path, video)

                if len(listdir_nohidden(video_path)) < 3:
                    # print(video_path)
                    remove_list.append(video)
                    continue

                self.video_path_list.append(video_path)
                self.data_label.append(data_dict[class_name])

            for audio in audio_list:
                if audio in remove_list:
                    print(audio)
                    continue
                audio_path = os.path.join(audio_class_path, audio)
                self.audio_path_list.append(audio_path)

        # print(len(self.data_label))

        # self.audio_path_list = self.audio_path_list[0:self.args.data_num]
        # self.video_path_list = self.video_path_list[0:self.args.data_num]
        # self.data_label = self.data_label[0:self.args.data_num]


        # audio_data = []
        # visual_data = []
        # label_data = []
        # count = torch.zeros(len(class_list))
        # for i in range(len(self.data_label)):
        #     label=self.data_label[i]
        #     if count[label] < self.args.data_num:
        #         audio_data.append(self.audio_path_list[i])
        #         visual_data.append(self.video_path_list[i])
        #         label_data.append(self.data_label[i])
        #         count[label] += 1

        # self.image = self.image[0:self.args.data_num]
        # self.label = self.label[0:self.args.data_num]
        # self.audio = self.audio[0:self.args.data_num]

        # self.video_path_list = visual_data
        # self.audio_path_list = audio_data
        # self.data_label = label_data

        # print("1",len(self.video_path_list))



    def __len__(self):
        # return 10000

        # if self.args.data_num < len(self.data_label):
        #
        #     return self.args.data_num
        # else:
        # print(len(self.data_label))
        return len(self.data_label)

    def __getitem__(self, idx):

        # audio
        sample, rate = librosa.load(self.audio_path_list[idx], sr=16000, mono=True)
        while len(sample) / rate < 10.:
            sample = np.tile(sample, 2)

        start_point = random.randint(a=0, b=rate * 5)
        new_sample = sample[start_point:start_point + rate * 5]
        new_sample[new_sample > 1.] = 1.
        new_sample[new_sample < -1.] = -1.

        spectrogram = librosa.stft(new_sample, n_fft=512, hop_length=256)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)
        spectrogram = np.transpose(spectrogram, (1, 0))
        # print(spectrogram.shape)

        # spectrogram=np.reshape(spectrogram,(spectrogram.shape[0]//2,spectrogram.shape[1]*2))

        spectrogram = np.transpose(spectrogram, (1, 0))
        # print(spectrogram.shape)

        spectrogram = np.resize(spectrogram, (224, 224))

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
        image_samples = listdir_nohidden(self.video_path_list[idx])
        # print(len(image_samples))
        select_index = np.random.choice(len(image_samples), size=self.args.use_video_frames, replace=False)
        select_index.sort()
        images = torch.zeros((self.args.use_video_frames, 3, 224, 224))
        for i in range(self.args.use_video_frames):
            try:
                img = Image.open(image_samples[i]).convert('RGB')
            except Exception as e:
                print(e)
                print(image_samples[i])
                continue

            bt = time.time()
            img = transform(img)
            et = time.time()
            # print(et-bt)
            images[i] = img

        images = torch.permute(images, (1, 0, 2, 3))

        # label
        label = self.data_label[idx]
        # print(label)

        return spectrogram, images, label


class KSDataset(nn.Module):
    def __init__(self, args, mode='train', data_path='./train_test_data/kinect_sound'):
        super().__init__()

        f = open('dataset/data/KineticSound/class.txt')
        data = f.readline()
        class_list = data.split(',')
        for i in range(len(class_list)):
            if " " in class_list[i]:
                class_name = class_list[i].split(" ")
                if class_name[0] == '':
                    class_name = class_name[1:len(class_name)]
                class_name = '_'.join(class_name)
                class_list[i] = class_name

        self.args = args

        # class_list=[class_list[0],class_list[1]]

        label = range(len(class_list))
        data_dict = zip(class_list, label)
        data_dict = dict(data_dict)

        # print(data_dict)

        self.mode = mode
        if self.mode == 'train':
            visual_data_path = os.path.join(data_path, 'visual', 'train_img/Image-01-FPS')
            audio_data_path = os.path.join(data_path, 'audio', 'train')
        elif self.mode == 'test':
            visual_data_path = os.path.join(data_path, 'visual', 'val_img/Image-01-FPS')
            audio_data_path = os.path.join(data_path, 'audio', 'test')

        self.data_label = []
        self.video_path_list = []
        self.audio_path_list = []

        remove_list = []  # 移除损坏视频

        # i=0
        for class_name in class_list:
            visual_class_path = os.path.join(visual_data_path, class_name)
            audio_class_path = os.path.join(audio_data_path, class_name)

            video_list = os.listdir(visual_class_path)
            video_list.sort()

            audio_list = os.listdir(audio_class_path)
            audio_list.sort()

            for video in video_list:
                # i+=1
                video_path = os.path.join(visual_class_path, video)

                if len(listdir_nohidden(video_path)) < 3:
                    # print(video_path)
                    remove_list.append(video)
                    continue

                self.video_path_list.append(video_path)
                self.data_label.append(data_dict[class_name])

            for audio in audio_list:
                if audio in remove_list:
                    print(audio)
                    continue
                audio_path = os.path.join(audio_class_path, audio)
                self.audio_path_list.append(audio_path)

        # print(len(self.data_label))

        # self.audio_path_list = self.audio_path_list[0:self.args.data_num]
        # self.video_path_list = self.video_path_list[0:self.args.data_num]
        # self.data_label = self.data_label[0:self.args.data_num]


        # audio_data = []
        # visual_data = []
        # label_data = []
        # count = torch.zeros(len(class_list))
        # for i in range(len(self.data_label)):
        #     label=self.data_label[i]
        #     if count[label] < self.args.data_num:
        #         audio_data.append(self.audio_path_list[i])
        #         visual_data.append(self.video_path_list[i])
        #         label_data.append(self.data_label[i])
        #         count[label] += 1

        # self.image = self.image[0:self.args.data_num]
        # self.label = self.label[0:self.args.data_num]
        # self.audio = self.audio[0:self.args.data_num]

        # self.video_path_list = visual_data
        # self.audio_path_list = audio_data
        # self.data_label = label_data

        # print("1",len(self.video_path_list))



    def __len__(self):
        # return 10000

        # if self.args.data_num < len(self.data_label):
        #
        #     return self.args.data_num
        # else:
        # print(len(self.data_label))
        return len(self.data_label)

    def __getitem__(self, idx):

        # audio
        sample, rate = librosa.load(self.audio_path_list[idx], sr=16000, mono=True)
        while len(sample) / rate < 10.:
            sample = np.tile(sample, 2)

        start_point = random.randint(a=0, b=rate * 5)
        new_sample = sample[start_point:start_point + rate * 5]
        new_sample[new_sample > 1.] = 1.
        new_sample[new_sample < -1.] = -1.

        spectrogram = librosa.stft(new_sample, n_fft=256, hop_length=128)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)
        spectrogram = np.transpose(spectrogram, (1, 0))
        # print(spectrogram.shape)

        # spectrogram=np.reshape(spectrogram,(spectrogram.shape[0]//2,spectrogram.shape[1]*2))

        spectrogram = np.transpose(spectrogram, (1, 0))
        # print(spectrogram.shape)

        # spectrogram = np.resize(spectrogram, (224, 224))
        # print(spectrogram.shape)

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
        image_samples = listdir_nohidden(self.video_path_list[idx])
        # print(len(image_samples))
        select_index = np.random.choice(len(image_samples), size=self.args.use_video_frames, replace=False)
        select_index.sort()
        images = torch.zeros((self.args.use_video_frames, 3, 224, 224))
        for i in range(self.args.use_video_frames):
            try:
                img = Image.open(image_samples[i]).convert('RGB')
            except Exception as e:
                print(e)
                print(image_samples[i])
                continue

            bt = time.time()
            img = transform(img)
            et = time.time()
            # print(et-bt)
            images[i] = img

        images = torch.permute(images, (1, 0, 2, 3))

        # label
        label = self.data_label[idx]
        # print(label)

        return spectrogram, images, label
