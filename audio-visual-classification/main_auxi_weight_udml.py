import argparse
import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.CramedDataset import CramedDataset
from dataset.KSDataset import KSDataset_Noise
from models.basic_model import AVClassifier_AUXI_UDML
from utils.utils import weight_init


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CREMAD', type=str,
                        choices=['CREMAD', 'KineticSound'],
                        help='release build only supports CREMAD and KineticSound')
    parser.add_argument('--modulation', default='OGM_GE', type=str,
                        choices=['Normal', 'OGM', 'OGM_GE'])
    parser.add_argument('--fusion_method', default='concat', type=str,
                        choices=['sum', 'concat', 'gated', 'film'])
    parser.add_argument('--fps', default=1, type=int)
    parser.add_argument('--num_frame', default=3, type=int, help='use how many frames for train')
    parser.add_argument('--audio_path', default='./train_test_data/CREMA-D/AudioWAV', type=str)
    parser.add_argument('--visual_path', default='./train_test_data/CREMA-D', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default='[70]', type=str, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')
    parser.add_argument('--ckpt_path', required=True, type=str, help='path to save trained models')
    parser.add_argument('--train', action='store_true', help='turn on train mode')
    parser.add_argument('--use_tensorboard', default=False, type=bool, help='whether to visualize')
    parser.add_argument('--tensorboard_path', type=str, help='path to save tensorboard logs')
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--gpu_ids', default='1', type=str, help='GPU ids')
    parser.add_argument('--pe', type=int, default=0)
    parser.add_argument('--modality', type=str, default='full')
    parser.add_argument('--beta', type=float, default=0)
    parser.add_argument('--backbone', type=str, default='resnet', choices=['resnet'])
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--drop', default=0, type=int)
    parser.add_argument('--cylcle_epoch', default=50, type=int)
    parser.add_argument('--audio_depend', default=32.0, type=float)
    parser.add_argument('--visual_depend', default=10.0, type=float)
    return parser.parse_args()


def get_feature_diversity(feature):
    feature = feature.view(feature.shape[0], feature.shape[1], -1)
    feature = feature.permute(0, 2, 1)
    feature = feature - torch.mean(feature, dim=2, keepdim=True)
    similarity = torch.bmm(feature, feature.permute(0, 2, 1))
    std = torch.std(feature, dim=2)
    std_matrix = torch.bmm(std.unsqueeze(dim=2), std.unsqueeze(dim=1))
    similarity = similarity / std_matrix
    norm = torch.norm(similarity, dim=(1, 2)) / (similarity.shape[1] ** 2)
    return torch.mean(norm)


def regurize(mul, std, target_var=2):
    variance_dul = std ** 2
    variance_dul = variance_dul.view(variance_dul.shape[0], -1)
    mul = mul.view(mul.shape[0], -1)
    target_var = torch.unsqueeze(target_var, dim=1).cuda()
    loss_kl = (
        (variance_dul / target_var)
        + (mul ** 2 / target_var)
        - torch.log((variance_dul + 1e-8) / target_var)
        - 1
    ) * 0.5
    loss_kl = torch.sum(loss_kl, dim=1)
    return torch.mean(loss_kl)


def train_epoch(args, epoch, model, device, dataloader, optimizer, scheduler):
    criterion = nn.CrossEntropyLoss()

    if scheduler is not None:
        scheduler.step(epoch)
    print(epoch, optimizer.param_groups[0]['lr'])

    model.train()
    print("Start training ... ")

    total_loss = 0
    total_loss_a = 0
    total_loss_v = 0
    total_a_diveristy = 0
    total_v_diveristy = 0
    total_a_re = 0
    total_v_re = 0
    similar_average = 0

    model.module.args.current_epoch = epoch

    for step, (spec, image, label, v_variance, a_variance) in enumerate(
        tqdm(dataloader, desc="Epoch {}/{}".format(epoch, args.epochs))
    ):
        spec = spec.to(device)
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        a, v, out, a_feature, v_feature, a_mul, a_std, v_mul, v_std, out_a, out_v, a_std_fc, v_std_fc, weight_a, weight_v = model(
            spec.unsqueeze(1).float(),
            image.float()
        )

        loss_v = criterion(out_v, label)
        loss_a = criterion(out_a, label)
        loss_f = criterion(out, label)

        calculate_a = torch.mean(torch.abs(out_a), 0).sum().cpu().detach()
        calculate_b = torch.mean(torch.abs(out_v), 0).sum().cpu().detach()
        model.module.args.audio_depend = calculate_a
        model.module.args.visual_depend = calculate_b

        loss_cls = loss_f + (loss_a + loss_v) * args.gamma
        a_diveristy = get_feature_diversity(a_feature)
        v_diveristy = get_feature_diversity(v_feature)

        if not isinstance(a_mul, int):
            regurize_a = regurize(a_mul, a_std, target_var=a_variance).cuda()
        else:
            regurize_a = torch.zeros(1).float().cuda()
            a_std = torch.zeros(1).float().cuda()

        if not isinstance(v_mul, int):
            if args.num_frame > 1:
                v_variance_kl = torch.repeat_interleave(v_variance, args.num_frame)
            else:
                v_variance_kl = v_variance
            regurize_v = regurize(v_mul, v_std, target_var=v_variance_kl).cuda()
        else:
            regurize_v = torch.zeros(1).float().cuda()
            v_std = torch.zeros(1).float().cuda()

        regurize_loss = regurize_a + regurize_v
        v_variance = torch.unsqueeze(v_variance.float(), dim=1).cuda()
        a_variance = torch.unsqueeze(a_variance.float(), dim=1).cuda()

        variance_fc_loss = F.mse_loss(a_std_fc, a_variance) + F.mse_loss(v_std_fc, v_variance)
        if variance_fc_loss == torch.inf:
            variance_fc_loss = torch.zeros(1).float().cuda()

        loss = loss_cls + regurize_loss * args.beta + variance_fc_loss * 0.1

        if step % 100 == 0:
            print("regurize_Loss:", regurize_loss.item(), "unimodal_loss:", (loss_a + loss_v).item(), "cls_loss:",
                  loss_cls.item(), "var_loss:", variance_fc_loss.item())
            print(weight_a, weight_v)
            print("calculate:", calculate_a, calculate_b)
            print("variance:", a_std.mean().item(), v_std.mean().item(), a_std_fc.mean().item(),
                  v_std_fc.mean().item(), a_variance.mean().item(), v_variance.mean().item())
            selected_rows_a = torch.index_select(model.module.fusion_module.fc_out.weight[:, :512], dim=0, index=label)
            selected_rows_v = torch.index_select(model.module.fusion_module.fc_out.weight[:, 512:], dim=0, index=label)
            distance_a = torch.abs(F.cosine_similarity(a, selected_rows_a, dim=1).mean())
            distance_v = torch.abs(F.cosine_similarity(v, selected_rows_v, dim=1).mean())
            print("distance:", distance_a.item(), distance_v.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=40, norm_type=2)

        audio_grad_sum = 0
        for p in model.module.audio_net.parameters():
            audio_grad_sum += torch.abs(p.grad).mean().item()

        visual_grad_sum = 0
        for p in model.module.visual_net.parameters():
            visual_grad_sum += torch.abs(p.grad).mean().item()

        if step % 100 == 0:
            print("grad:", audio_grad_sum, visual_grad_sum)

        with open('audio_visual_grad_vanilla.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([audio_grad_sum, visual_grad_sum])

        if args.modulation != 'Normal':
            audio_scale = audio_grad_sum
            visual_scale = visual_grad_sum

            for p in model.module.audio_net.parameters():
                p.grad = p.grad * visual_scale / audio_scale + torch.zeros_like(p.grad).normal_(
                    0, p.grad.std().item() + 1e-8
                )
            for p in model.module.visual_net.parameters():
                p.grad = p.grad * audio_scale / visual_scale + torch.zeros_like(p.grad).normal_(
                    0, p.grad.std().item() + 1e-8
                )

        optimizer.step()

        total_loss += loss.item()
        total_loss_a += loss_a.item()
        total_loss_v += loss_v.item()
        total_a_diveristy += a_diveristy.item()
        total_v_diveristy += v_diveristy.item()
        total_a_re += regurize_a.item()
        total_v_re += regurize_v.item()

    similar_average = similar_average / (step + 1)
    print("mse_diff:", similar_average)
    print(total_loss, len(dataloader))

    return (
        total_loss / len(dataloader),
        total_loss_a / len(dataloader),
        total_loss_v / len(dataloader),
        total_a_diveristy / len(dataloader),
        total_v_diveristy / len(dataloader),
        total_a_re / len(dataloader),
        total_v_re / len(dataloader),
    )


def valid(args, model, device, dataloader):
    softmax = nn.Softmax(dim=1)

    if args.dataset == 'KineticSound':
        n_classes = 34
    elif args.dataset == 'CREMAD':
        n_classes = 6
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

    model.module.args.drop = 0
    with torch.no_grad():
        model.eval()
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]

        for spec, image, label, a_variance, v_variance in dataloader:
            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)

            _, _, out, _, _, _, _, _, _, out_a, out_v, _, _, _, _ = model(
                spec.unsqueeze(1).float(),
                image.float()
            )

            prediction = softmax(out)
            pred_v = softmax(out_v)
            pred_a = softmax(out_a)

            for i in range(image.shape[0]):
                ma = np.argmax(prediction[i].cpu().data.numpy())
                v = np.argmax(pred_v[i].cpu().data.numpy())
                a = np.argmax(pred_a[i].cpu().data.numpy())
                num[label[i]] += 1.0

                if np.asarray(label[i].cpu()) == ma:
                    acc[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == v:
                    acc_v[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == a:
                    acc_a[label[i]] += 1.0

    return sum(acc) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num)


def build_datasets(args):
    if args.dataset == 'KineticSound':
        train_dataset_clean = KSDataset_Noise(args, mode='train', add_noise=False)
        train_dataset_noise = KSDataset_Noise(args, mode='train', add_noise=True)
        test_dataset = KSDataset_Noise(args, mode='test')
    elif args.dataset == 'CREMAD':
        train_dataset_clean = CramedDataset(args, mode='train', add_noise=False)
        train_dataset_noise = CramedDataset(args, mode='train', add_noise=True)
        test_dataset = CramedDataset(args, mode='test', add_noise=True)
    else:
        raise NotImplementedError(
            'Incorrect dataset name {}! Only support CREMAD and KineticSound in this release build!'.format(
                args.dataset
            )
        )

    return train_dataset_clean, train_dataset_noise, test_dataset


def build_optimizer(args, model):
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, eval(args.lr_decay_step), args.lr_decay_ratio)
    elif args.optimizer == 'AdaGrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.learning_rate)
        scheduler = None
    elif args.optimizer == 'Adam':
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
        scheduler = None
    else:
        raise ValueError

    return optimizer, scheduler


def save_checkpoint(args, model, epoch, acc):
    model_name = 'best_model_of_dataset_{}_{}_gamma_{}_pe_{}_beta{}_epoch_{}_acc_{}.pth'.format(
        args.dataset,
        args.modulation,
        args.gamma,
        args.pe,
        args.beta,
        epoch,
        acc
    )
    saved_dict = {
        'model': model.state_dict(),
        'audio_depend': float(model.module.args.audio_depend),
        'visual_depend': float(model.module.args.visual_depend),
        'beta': args.beta,
        'gamma': args.gamma,
    }
    save_dir = os.path.join(args.ckpt_path, model_name)
    torch.save(saved_dict, save_dir)
    return save_dir


def main():
    args = get_arguments()
    args.p = [0, 0]
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    gpu_ids = list(range(torch.cuda.device_count()))
    device = torch.device('cuda:0')

    model = AVClassifier_AUXI_UDML(args)
    model.apply(weight_init)
    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    model.cuda()

    optimizer, scheduler = build_optimizer(args, model)
    train_dataset_clean, train_dataset_noise, test_dataset = build_datasets(args)

    train_dataloader_clean = DataLoader(train_dataset_clean, batch_size=args.batch_size,
                                        shuffle=True, num_workers=32, pin_memory=True, drop_last=True)
    train_dataloader_noise = DataLoader(train_dataset_noise, batch_size=args.batch_size,
                                        shuffle=True, num_workers=32, pin_memory=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=32, pin_memory=True, drop_last=True)

    if not os.path.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path)

    log_path = os.path.join(args.ckpt_path, args.dataset + '_' + args.modality + '.csv')
    with open(log_path, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow([1000, 1000, 1000])

    if args.train:
        best_acc = 0.0

        for epoch in range(args.epochs):
            if epoch < args.cylcle_epoch:
                train_dataloader = train_dataloader_clean
            else:
                train_dataloader = train_dataloader_noise

            print('Epoch: {}: '.format(epoch))
            args.epoch_now = epoch

            if args.use_tensorboard:
                writer_path = os.path.join(args.tensorboard_path, args.dataset)
                if not os.path.exists(writer_path):
                    os.mkdir(writer_path)
                log_name = '{}_{}'.format(args.fusion_method, args.modulation)
                writer = SummaryWriter(os.path.join(writer_path, log_name))

                batch_loss, batch_loss_a, batch_loss_v, a_diveristy, v_diveristy, a_re, v_re = train_epoch(
                    args, epoch, model, device, train_dataloader, optimizer, scheduler
                )
                acc, acc_a, acc_v = valid(args, model, device, test_dataloader)

                writer.add_scalars('Loss', {
                    'Total Loss': batch_loss,
                    'Audio Loss': batch_loss_a,
                    'Visual Loss': batch_loss_v
                }, epoch)
                writer.add_scalars('Evaluation', {
                    'Total Accuracy': acc,
                    'Audio Accuracy': acc_a,
                    'Visual Accuracy': acc_v
                }, epoch)
            else:
                batch_loss, batch_loss_a, batch_loss_v, a_diveristy, v_diveristy, a_re, v_re = train_epoch(
                    args=args,
                    epoch=epoch,
                    model=model,
                    device=device,
                    dataloader=train_dataloader,
                    optimizer=optimizer,
                    scheduler=scheduler
                )
                acc, acc_a, acc_v = valid(args, model, device, test_dataloader)

                print(11111111111)
                with open(log_path, 'a+', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=",")
                    writer.writerow([acc, acc_a, acc_v])

            if acc > best_acc and epoch > args.cylcle_epoch:
                best_acc = float(acc)
                save_dir = save_checkpoint(args, model, epoch, acc)
                print('The best model has been saved at {}.'.format(save_dir))
                print("Loss: {:.3f}, Acc: {:.3f}".format(batch_loss, acc))
                print("Audio Acc: {:.3f}锛?Visual Acc: {:.3f} ".format(acc_a, acc_v))
                print("Audio similar: {:.3f}锛?Visual similar: {:.3f} ".format(a_diveristy, v_diveristy))
                print("Audio regurize: {:.3f}锛?Visual regurize: {:.3f} ".format(a_re, v_re))
            else:
                print("Loss: {:.3f}, Acc: {:.3f}, Best Acc: {:.3f}".format(batch_loss, acc, best_acc))
                print("Audio Acc: {:.3f}锛?Visual Acc: {:.3f} ".format(acc_a, acc_v))
                print("Audio similar: {:.3f}锛?Visual similar: {:.3f} ".format(a_diveristy, v_diveristy))
                print("Audio regurize: {:.3f}锛?Visual regurize: {:.3f} ".format(a_re, v_re))
    else:
        loaded_dict = torch.load(
            "./results/cramed/udml/best_model_of_dataset_CREMAD_Normal_gamma_4.0_pe_1_beta1e-05_optimizer_sgd_modulate_starts_0_ends_50_epoch_91_acc_0.6605113636363636.pth"
        )
        model.load_state_dict(loaded_dict['model'])
        print('Trained model loaded!')

        acc, acc_a, acc_v = valid(args, model, device, test_dataloader)
        print('Accuracy: {}, accuracy_a: {}, accuracy_v: {}'.format(acc, acc_a, acc_v))


if __name__ == "__main__":
    main()

