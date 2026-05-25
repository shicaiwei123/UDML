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

        self.a_variance_list=[]
        self.v_variance_list=[]
        for i in range(len(self.data_label)):
            a = np.float(np.random.randint(low=1, high=12))
            v=np.float(np.random.randint(low=1, high=12))
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
        # return 10000

        return len(self.data_label)

    def __getitem__(self, idx):

        
        if self.add_noise:
            visual_variance = self.v_variance_list[idx]
            audio_variance = self.a_variance_list[idx]
        else:
            visual_variance=1
            audio_variance=1
        
        visual_noise_process=AddGaussianNoise(variance=visual_variance)
        audio_noise_process=AddGaussianNoise_spec(variance=audio_variance)
        
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

        # print(np.mean(spectrogram))

        if self.mode=='train':
            spectrogram = audio_noise_process(spectrogram)
        else:
            audio_noise_process=AddGaussianNoise_spec(variance=1)
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
                AddGaussianNoise(variance=1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

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