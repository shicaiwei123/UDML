import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import resnet18, resnet50, resnet18_weight
from .fusion_modules import SumFusion, ConcatFusion, FiLM, GatedFusion, ConcatFusion_Swin, ConcatFusion_AUXI, \
    GatedFusion_AUXI, SumFusion_AUXI, FiLM_AUXI, ShareWeightFusion_AUXI
import numpy as np


def modality_drop(x_rgb, x_depth, p, args=None):
    modality_combination = [[1, 0], [0, 1], [1, 1]]
    index_list = [x for x in range(3)]

    if p == [0, 0]:
        p = []

        # for i in range(x_rgb.shape[0]):
        #     index = random.randint(0, 6)
        #     p.append(modality_combination[index])
        #     if 'model_arch_index' in args.writer_dicts.keys():
        #         args.writer_dicts['model_arch_index'].write(str(index) + " ")
        prob = np.array((1 / 3, 1 / 3, 1 / 3))
        for i in range(x_rgb.shape[0]):
            index = np.random.choice(index_list, size=1, replace=True, p=prob)[0]
            p.append(modality_combination[index])
            # if 'model_arch_index' in args.writer_dicts.keys():
            #     args.writer_dicts['model_arch_index'].write(str(index) + " ")

        # if [0, 1] not in p:
        #     p[0] = [0, 1]
        p = np.array(p)
        p = torch.from_numpy(p)
        p = torch.unsqueeze(p, 2)
        p = torch.unsqueeze(p, 3)
        p = torch.unsqueeze(p, 4)

    else:
        p = p
        # print(p)
        p = [p * x_rgb.shape[0]]
        # print(p)
        p = np.array(p).reshape(x_rgb.shape[0], 2)
        p = torch.from_numpy(p)
        p = torch.unsqueeze(p, 2)
        p = torch.unsqueeze(p, 3)
        p = torch.unsqueeze(p, 4)

    p = p.float().cuda()

    x_rgb = x_rgb * p[:, 0]

    if x_rgb.shape[0] != x_depth.shape[0]:
        pv = torch.repeat_interleave(p, args.num_frame, dim=0)
        # print(pv.shape)
        x_depth = x_depth * pv[:, 1]
    else:
        x_depth = x_depth * p[:, 1]

    return x_rgb, x_depth, p


class MultiHeadSelfAttention(nn.Module):
    """Self-attention module by Lin, Zhouhan, et al. ICLR 2017"""

    def __init__(self, n_head, d_in, d_hidden):
        super(MultiHeadSelfAttention, self).__init__()

        self.n_head = n_head
        self.w_1 = nn.Linear(d_in, d_hidden)
        self.w_2 = nn.Linear(d_hidden, n_head)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    #     self.init_weights()
    #
    # def init_weights(self):
    #     nn.init.xavier_uniform_(self.w_1.weight)
    #     nn.init.xavier_uniform_(self.w_2.weight)

    def forward(self, x, mask=None):
        # This expects input x to be of size (b x seqlen x d_feat)
        attn = self.w_2(self.tanh(self.w_1(x)))
        if mask is not None:
            mask = mask.repeat(self.n_head, 1, 1).permute(1, 2, 0)
            attn.masked_fill_(mask, -np.inf)
        attn = self.softmax(attn)

        output = torch.bmm(attn.transpose(1, 2), x)
        if output.shape[1] == 1:
            output = output.squeeze(1)
        return output, attn


class PCME(nn.Module):
    def __init__(self, d_in, d_out, d_h):
        super().__init__()

        self.attention = MultiHeadSelfAttention(1, d_in, d_h)

        self.fc = nn.Linear(d_in, d_out)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

        self.fc2 = nn.Linear(d_in, d_out)
        self.embed_dim = d_in

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, out, x, pad_mask=None):
        residual, attn = self.attention(x, pad_mask)

        fc_out = self.fc2(out)
        out = self.fc(residual) + fc_out

        return out


class AVClassifier(nn.Module):
    def __init__(self, args):
        super(AVClassifier, self).__init__()

        fusion = args.fusion_method
        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 34
        elif args.dataset == 'kinect400':
            n_classes = 400
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        if fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            if args.dataset == 'kinect400':
                self.fusion_module = ConcatFusion(output_dim=n_classes, input_dim=1024)
            else:
                self.fusion_module = ConcatFusion(output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion(output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        if args.modality == 'full':
            self.audio_net = resnet18(modality='audio', args=args)
            self.visual_net = resnet18(modality='visual', args=args)

        if args.modality == 'visual':
            if args.dataset == 'kinect400':
                print("resnet50")
                self.visual_net = resnet18(modality='visual', args=args)
                self.visual_classifier = nn.Linear(512, n_classes)
            else:
                self.visual_net = resnet18(modality='visual', args=args)
                self.visual_classifier = nn.Linear(512, n_classes)
        if args.modality == 'audio':
            if args.dataset == 'kinect400':
                print("resnet50")
                self.audio_net = resnet18(modality='audio', args=args)
                self.audio_classifier = nn.Linear(512, n_classes)
            else:
                self.audio_net = resnet18(modality='audio', args=args)
                self.audio_classifier = nn.Linear(512, n_classes)
        self.pe = args.pe
        self.modality = args.modality
        self.args = args

        self.unimodal_fc = nn.Linear(512, n_classes)

        # self.audio_mu = nn.Linear(512, 512)
        # self.audio_logval = PCME(512, 512, 256)
        # self.visual_mu = nn.Linear(512, 512)
        # self.visual_logval = PCME(512, 512, 256)




    def forward(self, audio, visual):

        if self.modality == 'full':

            if self.pe:
                a, a_mul, a_std = self.audio_net(audio)  # only feature
                v, v_mul, v_std = self.visual_net(visual)

                if self.args.drop:
                    a, v, p = modality_drop(a, v, self.args.p, args=self.args)

                a_feature = a
                v_feature = v

                (_, C, H, W) = v.size()
                B = a.size()[0]
                v = v.view(B, -1, C, H, W)
                v = v.permute(0, 2, 1, 3, 4)

                a = F.adaptive_avg_pool2d(a, 1)
                v = F.adaptive_avg_pool3d(v, 1)

                a = torch.flatten(a, 1)
                v = torch.flatten(v, 1)

                # v=v*0

                a_out = self.unimodal_fc(a)
                v_out = self.unimodal_fc(v)

                a, v, out = self.fusion_module(a, v)  # av 是原来的，out是融合结果

                return a, v, out, a_feature, v_feature, a_mul, a_std, v_mul, v_std, a_out, v_out
            else:
                a = self.audio_net(audio)  # only feature
                v = self.visual_net(visual)

                a_feature = a
                v_feature = v

                (_, C, H, W) = v.size()
                B = a.size()[0]
                v = v.view(B, -1, C, H, W)
                v = v.permute(0, 2, 1, 3, 4)

                a = F.adaptive_avg_pool2d(a, 1)
                v = F.adaptive_avg_pool3d(v, 1)

                a = torch.flatten(a, 1)
                v = torch.flatten(v, 1)

                a_out = self.unimodal_fc(a)
                v_out = self.unimodal_fc(v)

                a, v, out = self.fusion_module(a, v)  # av 是原来的，out是融合结果

                return a, v, out, a_feature, v_feature, 0, 0, 0, 0, a_out, v_out
        elif self.modality == 'visual':

            if self.pe:
                v, v_mul, v_std = self.visual_net(visual)

                v_feature = v

                (_, C, H, W) = v.size()
                B = self.args.batch_size
                v = v.view(B, -1, C, H, W)
                v = v.permute(0, 2, 1, 3, 4)

                v = F.adaptive_avg_pool3d(v, 1)

                v = torch.flatten(v, 1)

                out = self.visual_classifier(v)

                a = torch.zeros_like(v)

                return a, v, out, v_feature, v_feature, 0, 0, v_mul, v_std


            else:

                v = self.visual_net(visual)

                v_feature = v

                (_, C, H, W) = v.size()
                B = self.args.batch_size
                v = v.view(B, -1, C, H, W)

                v = v.permute(0, 2, 1, 3, 4)

                v = F.adaptive_avg_pool3d(v, 1)

                v = torch.flatten(v, 1)

                out = self.visual_classifier(v)

                a = torch.zeros_like(v)

                # print(11111111111111)

                return a, v, out, v_feature, v_feature, 0, 0, 0, 0

        elif self.modality == 'audio':

            if self.pe:
                a, a_mul, a_std = self.audio_net(audio)  # only feature

                a_feature = a
                a = F.adaptive_avg_pool2d(a, 1)

                a = torch.flatten(a, 1)
                out = self.audio_classifier(a)
                v = torch.zeros_like(a)

                return a, v, out, a_feature, a_feature, a_mul, a_std, 0, 0

            else:

                a = self.audio_net(audio)  # only feature
                a_feature = a

                a = F.adaptive_avg_pool2d(a, 1)

                a = torch.flatten(a, 1)

                out = self.audio_classifier(a)
                v = torch.zeros_like(a)

                return a, v, out, a_feature, a_feature, 0, 0, 0, 0
        else:
            return 0, 0, 0


class AVClassifier_AUXI(nn.Module):
    def __init__(self, args):
        super(AVClassifier_AUXI, self).__init__()

        fusion = args.fusion_method
        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 34
        elif args.dataset == 'kinect400':
            n_classes = 400
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        if fusion == 'sum':
            self.fusion_module = SumFusion_AUXI(output_dim=n_classes)
        elif fusion == 'concat':
            if args.dataset == 'kinect400':
                self.fusion_module = ConcatFusion_AUXI(output_dim=n_classes, input_dim=1024)
            else:
                self.fusion_module = ConcatFusion_AUXI(output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM_AUXI(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion_AUXI(output_dim=n_classes, x_gate=True)
        elif fusion == 'share':
            self.fusion_module = ShareWeightFusion_AUXI(output_dim=n_classes)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        if args.modality == 'full':
            self.audio_net = resnet18(modality='audio', args=args)
            self.visual_net = resnet18(modality='visual', args=args)

        if args.modality == 'visual':
            if args.dataset == 'kinect400':
                print("resnet50")
                self.visual_net = resnet18(modality='visual', args=args)
                self.visual_classifier = nn.Linear(512, n_classes)
            else:
                self.visual_net = resnet18(modality='visual', args=args)
                self.visual_classifier = nn.Linear(512, n_classes)
        if args.modality == 'audio':
            if args.dataset == 'kinect400':
                print("resnet50")
                self.audio_net = resnet18(modality='audio', args=args)
                self.audio_classifier = nn.Linear(512, n_classes)
            else:
                self.audio_net = resnet18(modality='audio', args=args)
                self.audio_classifier = nn.Linear(512, n_classes)
        self.pe = args.pe
        self.modality = args.modality
        self.args = args

        # self.audio_mu = nn.Linear(512, 512)
        # self.audio_logval = PCME(512, 512, 256)
        # self.visual_mu = nn.Linear(512, 512)
        # self.visual_logval = PCME(512, 512, 256)

        self.visual_variance_estimator=nn.Sequential(nn.Linear(512,256),nn.Dropout(0.1),nn.Linear(256,1))
        self.audio_variance_estimator=nn.Sequential(nn.Linear(512,256),nn.Dropout(0.1),nn.Linear(256,1))


    def forward(self, audio, visual):

        if self.modality == 'full':

            if self.pe:
                a, a_mul, a_std = self.audio_net(audio)  # only feature
                v, v_mul, v_std = self.visual_net(visual)

                if self.args.drop:
                    a, v, p = modality_drop(a, v, self.args.p, args=self.args)

                a_feature = a
                v_feature = v

                (_, C, H, W) = v.size()
                B = a.size()[0]
                v = v.view(B, -1, C, H, W)
                v = v.permute(0, 2, 1, 3, 4)

                a = F.adaptive_avg_pool2d(a, 1)
                v = F.adaptive_avg_pool3d(v, 1)

                a = torch.flatten(a, 1)
                v = torch.flatten(v, 1)



                v_std_in = v_std.view(B, -1, C, H, W)
                v_std_in = v_std_in.permute(0, 2, 1, 3, 4)

                a_std_in = F.adaptive_avg_pool2d(a_std, 1)
                v_std_in = F.adaptive_avg_pool3d(v_std_in, 1)

                a_std_in = torch.flatten(a_std_in, 1)
                v_std_in = torch.flatten(v_std_in, 1)



                a_std_fc=self.audio_variance_estimator(a_std_in.detach())
                v_std_fc=self.visual_variance_estimator(v_std_in.detach())

                a_std_fc = (a_std_fc * 0.5).exp()
                v_std_fc = (v_std_fc * 0.5).exp()

                if self.training:
                    if self.args.current_epoch<self.args.cylcle_epoch+10:
                        weight_a,weight_v=1,1
                    else:
                        weight_a,weight_v=2*v_std_fc**2/(v_std_fc**2+a_std_fc**2),2*a_std_fc**2/(v_std_fc**2+a_std_fc**2)

                        # teaching force
                        # a_varinace_label=torch.unsqueeze(a_varinace_label.float(),dim=1).cuda()
                        # v_varinace_label=torch.unsqueeze(v_varinace_label.float(),dim=1).cuda()
                        # weight_a,weight_v=2*v_varinace_label**2/(v_varinace_label**2+a_varinace_label**2),2*a_varinace_label**2/(v_varinace_label**2+a_varinace_label**2)
                else:
                    weight_a,weight_v=2*v_std_fc**2/(v_std_fc**2+a_std_fc**2),2*a_std_fc**2/(v_std_fc**2+a_std_fc**2)
                # # a_out=self.unimodal_fc(a)
                # # v_out=self.unimodal_fc(v)
                # weight_a,weight_v=1,1
                # print(weight_a,weight_v)
                a_out, v_out, out = self.fusion_module(a+weight_a*a, v+weight_v*v)  # av 是原来的，out是融合结果

                return a, v, out, a_feature, v_feature, a_mul, a_std, v_mul, v_std, a_out, v_out,a_std_fc,v_std_fc
            else:
                a = self.audio_net(audio)  # only feature
                v = self.visual_net(visual)

                a_feature = a
                v_feature = v

                (_, C, H, W) = v.size()
                B = a.size()[0]
                v = v.view(B, -1, C, H, W)
                v = v.permute(0, 2, 1, 3, 4)

                a = F.adaptive_avg_pool2d(a, 1)
                v = F.adaptive_avg_pool3d(v, 1)

                a = torch.flatten(a, 1)
                v = torch.flatten(v, 1)



                # v_std_in = v_std.view(B, -1, C, H, W)
                # v_std_in = v_std_in.permute(0, 2, 1, 3, 4)

                # a_std_in = F.adaptive_avg_pool2d(a_std, 1)
                # v_std_in = F.adaptive_avg_pool3d(v_std_in, 1)

                # a_std_in = torch.flatten(a_std_in, 1)
                # v_std_in = torch.flatten(v_std_in, 1)



                a_std_fc=self.audio_variance_estimator(a.detach())
                v_std_fc=self.visual_variance_estimator(v.detach())

                a_std_fc = (a_std_fc * 0.5).exp()
                v_std_fc = (v_std_fc * 0.5).exp()

                if self.training:
                    if self.args.current_epoch<self.args.cylcle_epoch+10:
                        weight_a,weight_v=1,1
                    else:
                        weight_a,weight_v=2*v_std_fc**2/(v_std_fc**2+a_std_fc**2),2*a_std_fc**2/(v_std_fc**2+a_std_fc**2)

                        # teaching force
                        # a_varinace_label=torch.unsqueeze(a_varinace_label.float(),dim=1).cuda()
                        # v_varinace_label=torch.unsqueeze(v_varinace_label.float(),dim=1).cuda()
                        # weight_a,weight_v=2*v_varinace_label**2/(v_varinace_label**2+a_varinace_label**2),2*a_varinace_label**2/(v_varinace_label**2+a_varinace_label**2)
                else:
                    weight_a,weight_v=2*v_std_fc**2/(v_std_fc**2+a_std_fc**2),2*a_std_fc**2/(v_std_fc**2+a_std_fc**2)
                # # a_out=self.unimodal_fc(a)
                # # v_out=self.unimodal_fc(v)
                # weight_a,weight_v=1,1
                # print(weight_a,weight_v)
                a_out, v_out, out = self.fusion_module(a+weight_a*a, v+weight_v*v)  # av 是原来的，out是融合结果

                return a, v, out, a_feature, v_feature, 0, 0, 0, 0, a_out, v_out,a_std_fc,v_std_fc

                # return a, v, out, a_feature, v_feature, 0, 0, 0, 0, a, v
        elif self.modality == 'visual':

            if self.pe:
                v, v_mul, v_std = self.visual_net(visual)

                v_feature = v

                (_, C, H, W) = v.size()
                B = self.args.batch_size
                v = v.view(B, -1, C, H, W)
                v = v.permute(0, 2, 1, 3, 4)

                v = F.adaptive_avg_pool3d(v, 1)

                v = torch.flatten(v, 1)

                out = self.visual_classifier(v)

                a = torch.zeros_like(v)

                return a, v, out, v_feature, v_feature, 0, 0, v_mul, v_std


            else:

                v = self.visual_net(visual)

                v_feature = v

                (_, C, H, W) = v.size()
                B = self.args.batch_size
                v = v.view(B, -1, C, H, W)

                v = v.permute(0, 2, 1, 3, 4)

                v = F.adaptive_avg_pool3d(v, 1)

                v = torch.flatten(v, 1)

                out = self.visual_classifier(v)

                a = torch.zeros_like(v)

                # print(11111111111111)

                return a, v, out, v_feature, v_feature, 0, 0, 0, 0

        elif self.modality == 'audio':

            if self.pe:
                a, a_mul, a_std = self.audio_net(audio)  # only feature

                a_feature = a
                a = F.adaptive_avg_pool2d(a, 1)

                a = torch.flatten(a, 1)
                out = self.audio_classifier(a)
                v = torch.zeros_like(a)

                return a, v, out, a_feature, a_feature, a_mul, a_std, 0, 0

            else:

                a = self.audio_net(audio)  # only feature
                a_feature = a

                a = F.adaptive_avg_pool2d(a, 1)

                a = torch.flatten(a, 1)

                out = self.audio_classifier(a)
                v = torch.zeros_like(a)

                return a, v, out, a_feature, a_feature, 0, 0, 0, 0
        else:
            return 0, 0, 0



class AVClassifier_AUXI_UDML_Center(nn.Module):
    def __init__(self, args):
        super(AVClassifier_AUXI_UDML_Center, self).__init__()

        fusion = args.fusion_method
        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 34
        elif args.dataset == 'kinect400':
            n_classes = 400
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        if fusion == 'sum':
            self.fusion_module = SumFusion_AUXI(output_dim=n_classes)
        elif fusion == 'concat':
            if args.dataset == 'kinect400':
                self.fusion_module = ConcatFusion_AUXI(output_dim=n_classes, input_dim=1024)
            else:
                self.fusion_module = ConcatFusion_AUXI(output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM_AUXI(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion_AUXI(output_dim=n_classes, x_gate=True)
        elif fusion == 'share':
            self.fusion_module = ShareWeightFusion_AUXI(output_dim=n_classes)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        if args.modality == 'full':
            self.audio_net = resnet18_weight(modality='audio', args=args)
            self.visual_net = resnet18_weight(modality='visual', args=args)

        if args.modality == 'visual':
            if args.dataset == 'kinect400':
                print("resnet50")
                self.visual_net = resnet18_weight(modality='visual', args=args)
                self.visual_classifier = nn.Linear(512, n_classes)
            else:
                self.visual_net = resnet18_weight(modality='visual', args=args)
                self.visual_classifier = nn.Linear(512, n_classes)
        if args.modality == 'audio':
            if args.dataset == 'kinect400':
                print("resnet50")
                self.audio_net = resnet18_weight(modality='audio', args=args)
                self.audio_classifier = nn.Linear(512, n_classes)
            else:
                self.audio_net = resnet18_weight(modality='audio', args=args)
                self.audio_classifier = nn.Linear(512, n_classes)
        self.pe = args.pe
        self.modality = args.modality
        self.args = args

        # self.audio_mu = nn.Linear(512, 512)
        # self.audio_logval = PCME(512, 512, 256)
        # self.visual_mu = nn.Linear(512, 512)
        # self.visual_logval = PCME(512, 512, 256)

        self.visual_variance_estimator=nn.Sequential(nn.Linear(512,256),nn.Dropout(0.1),nn.Linear(256,1))
        self.audio_variance_estimator=nn.Sequential(nn.Linear(512,256),nn.Dropout(0.1),nn.Linear(256,1))


    def forward(self, audio, visual,label):

        if self.modality == 'full':

            if self.pe:
                a, a_mul, a_std,a_center = self.audio_net(audio,label)  # only feature
                v, v_mul, v_std,v_center = self.visual_net(visual,label)

                if self.args.drop:
                    a, v, p = modality_drop(a, v, self.args.p, args=self.args)

                a_feature = a
                v_feature = v

                (_, C, H, W) = v.size()
                B = a.size()[0]
                v = v.view(B, -1, C, H, W)
                v = v.permute(0, 2, 1, 3, 4)

                a = F.adaptive_avg_pool2d(a, 1)
                v = F.adaptive_avg_pool3d(v, 1)

                a = torch.flatten(a, 1)
                v = torch.flatten(v, 1)



                v_std_in = v_std.view(B, -1, C, H, W)
                v_std_in = v_std_in.permute(0, 2, 1, 3, 4)

                a_std_in = F.adaptive_avg_pool2d(a_std, 1)
                v_std_in = F.adaptive_avg_pool3d(v_std_in, 1)

                a_std_in = torch.flatten(a_std_in, 1)
                v_std_in = torch.flatten(v_std_in, 1)



                a_std_fc=self.audio_variance_estimator(a_std_in.detach())
                v_std_fc=self.visual_variance_estimator(v_std_in.detach())

                a_std_fc = (a_std_fc * 0.5).exp()
                v_std_fc = (v_std_fc * 0.5).exp()

                if self.training:
                    if self.args.current_epoch<self.args.cylcle_epoch+10:
                        target_weight_a,target_weight_v=1,1
                    else:
                        target_weight_a,target_weight_v=2*v_std_fc**2/(v_std_fc**2+a_std_fc**2),2*a_std_fc**2/(v_std_fc**2+a_std_fc**2)

                        # teaching force
                        # a_varinace_label=torch.unsqueeze(a_varinace_label.float(),dim=1).cuda()
                        # v_varinace_label=torch.unsqueeze(v_varinace_label.float(),dim=1).cuda()
                        # weight_a,weight_v=2*v_varinace_label**2/(v_varinace_label**2+a_varinace_label**2),2*a_varinace_label**2/(v_varinace_label**2+a_varinace_label**2)
                else:
                    target_weight_a,target_weight_v=2*v_std_fc**2/(v_std_fc**2+a_std_fc**2),2*a_std_fc**2/(v_std_fc**2+a_std_fc**2)


                # print(target_weight_a.shape)
                weight_a=target_weight_a/self.args.audio_depend
                weight_v=target_weight_v/self.args.visual_depend

                weight_a,weight_v=2*weight_a/(weight_a+weight_v),2*weight_v/(weight_a+weight_v)
                
                # print(weight_a,weight_v)


                a_out, v_out, out = self.fusion_module(a+weight_a*a, v+weight_v*v)  # av 是原来的，out是融合结果

                return a, v, out, a_feature, v_feature, a_mul, a_std, v_mul, v_std, a_out, v_out,a_std_fc,v_std_fc,weight_a,weight_v,a_center,v_center
            else:
                a = self.audio_net(audio)  # only feature
                v = self.visual_net(visual)

                a_feature = a
                v_feature = v

                (_, C, H, W) = v.size()
                B = a.size()[0]
                v = v.view(B, -1, C, H, W)
                v = v.permute(0, 2, 1, 3, 4)

                a = F.adaptive_avg_pool2d(a, 1)
                v = F.adaptive_avg_pool3d(v, 1)

                a = torch.flatten(a, 1)
                v = torch.flatten(v, 1)



                # v_std_in = v_std.view(B, -1, C, H, W)
                # v_std_in = v_std_in.permute(0, 2, 1, 3, 4)

                # a_std_in = F.adaptive_avg_pool2d(a_std, 1)
                # v_std_in = F.adaptive_avg_pool3d(v_std_in, 1)

                # a_std_in = torch.flatten(a_std_in, 1)
                # v_std_in = torch.flatten(v_std_in, 1)



                a_std_fc=self.audio_variance_estimator(a.detach())
                v_std_fc=self.visual_variance_estimator(v.detach())

                a_std_fc = (a_std_fc * 0.5).exp()
                v_std_fc = (v_std_fc * 0.5).exp()

                if self.training:
                    if self.args.current_epoch<self.args.cylcle_epoch+10:
                        weight_a,weight_v=1,1
                    else:
                        weight_a,weight_v=2*v_std_fc**2/(v_std_fc**2+a_std_fc**2),2*a_std_fc**2/(v_std_fc**2+a_std_fc**2)

                        # teaching force
                        # a_varinace_label=torch.unsqueeze(a_varinace_label.float(),dim=1).cuda()
                        # v_varinace_label=torch.unsqueeze(v_varinace_label.float(),dim=1).cuda()
                        # weight_a,weight_v=2*v_varinace_label**2/(v_varinace_label**2+a_varinace_label**2),2*a_varinace_label**2/(v_varinace_label**2+a_varinace_label**2)
                else:
                    weight_a,weight_v=2*v_std_fc**2/(v_std_fc**2+a_std_fc**2),2*a_std_fc**2/(v_std_fc**2+a_std_fc**2)
                # # a_out=self.unimodal_fc(a)
                # # v_out=self.unimodal_fc(v)
                # weight_a,weight_v=1,1
                # print(weight_a,weight_v)
                a_out, v_out, out = self.fusion_module(a+weight_a*a, v+weight_v*v)  # av 是原来的，out是融合结果

                return a, v, out, a_feature, v_feature, 0, 0, 0, 0, a_out, v_out,a_std_fc,v_std_fc

                # return a, v, out, a_feature, v_feature, 0, 0, 0, 0, a, v
        elif self.modality == 'visual':

            if self.pe:
                v, v_mul, v_std = self.visual_net(visual)

                v_feature = v

                (_, C, H, W) = v.size()
                B = self.args.batch_size
                v = v.view(B, -1, C, H, W)
                v = v.permute(0, 2, 1, 3, 4)

                v = F.adaptive_avg_pool3d(v, 1)

                v = torch.flatten(v, 1)

                out = self.visual_classifier(v)

                a = torch.zeros_like(v)

                return a, v, out, v_feature, v_feature, 0, 0, v_mul, v_std


            else:

                v = self.visual_net(visual)

                v_feature = v

                (_, C, H, W) = v.size()
                B = self.args.batch_size
                v = v.view(B, -1, C, H, W)

                v = v.permute(0, 2, 1, 3, 4)

                v = F.adaptive_avg_pool3d(v, 1)

                v = torch.flatten(v, 1)

                out = self.visual_classifier(v)

                a = torch.zeros_like(v)

                # print(11111111111111)

                return a, v, out, v_feature, v_feature, 0, 0, 0, 0

        elif self.modality == 'audio':

            if self.pe:
                a, a_mul, a_std = self.audio_net(audio)  # only feature

                a_feature = a
                a = F.adaptive_avg_pool2d(a, 1)

                a = torch.flatten(a, 1)
                out = self.audio_classifier(a)
                v = torch.zeros_like(a)

                return a, v, out, a_feature, a_feature, a_mul, a_std, 0, 0

            else:

                a = self.audio_net(audio)  # only feature
                a_feature = a

                a = F.adaptive_avg_pool2d(a, 1)

                a = torch.flatten(a, 1)

                out = self.audio_classifier(a)
                v = torch.zeros_like(a)

                return a, v, out, a_feature, a_feature, 0, 0, 0, 0
        else:
            return 0, 0, 0
        




class AVClassifier_AUXI_UDML(nn.Module):
    def __init__(self, args):
        super(AVClassifier_AUXI_UDML, self).__init__()

        fusion = args.fusion_method
        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 34
        elif args.dataset == 'kinect400':
            n_classes = 400
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        if fusion == 'sum':
            self.fusion_module = SumFusion_AUXI(output_dim=n_classes)
        elif fusion == 'concat':
            if args.dataset == 'kinect400':
                self.fusion_module = ConcatFusion_AUXI(output_dim=n_classes, input_dim=1024)
            else:
                self.fusion_module = ConcatFusion_AUXI(output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM_AUXI(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion_AUXI(output_dim=n_classes, x_gate=True)
        elif fusion == 'share':
            self.fusion_module = ShareWeightFusion_AUXI(output_dim=n_classes)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        if args.modality == 'full':
            self.audio_net = resnet18(modality='audio', args=args)
            self.visual_net = resnet18(modality='visual', args=args)

        if args.modality == 'visual':
            if args.dataset == 'kinect400':
                print("resnet50")
                self.visual_net = resnet18(modality='visual', args=args)
                self.visual_classifier = nn.Linear(512, n_classes)
            else:
                self.visual_net = resnet18(modality='visual', args=args)
                self.visual_classifier = nn.Linear(512, n_classes)
        if args.modality == 'audio':
            if args.dataset == 'kinect400':
                print("resnet50")
                self.audio_net = resnet18(modality='audio', args=args)
                self.audio_classifier = nn.Linear(512, n_classes)
            else:
                self.audio_net = resnet18(modality='audio', args=args)
                self.audio_classifier = nn.Linear(512, n_classes)
        self.pe = args.pe
        self.modality = args.modality
        self.args = args

        # self.audio_mu = nn.Linear(512, 512)
        # self.audio_logval = PCME(512, 512, 256)
        # self.visual_mu = nn.Linear(512, 512)
        # self.visual_logval = PCME(512, 512, 256)

        self.visual_variance_estimator=nn.Sequential(nn.Linear(512,256),nn.Dropout(0.1),nn.Linear(256,1))
        self.audio_variance_estimator=nn.Sequential(nn.Linear(512,256),nn.Dropout(0.1),nn.Linear(256,1))


    def forward(self, audio, visual):

        if self.modality == 'full':

            if self.pe:
                a, a_mul, a_std = self.audio_net(audio)  # only feature
                v, v_mul, v_std = self.visual_net(visual)

                if self.args.drop:
                    a, v, p = modality_drop(a, v, self.args.p, args=self.args)

                a_feature = a
                v_feature = v

                (_, C, H, W) = v.size()
                B = a.size()[0]
                v = v.view(B, -1, C, H, W)
                v = v.permute(0, 2, 1, 3, 4)

                a = F.adaptive_avg_pool2d(a, 1)
                v = F.adaptive_avg_pool3d(v, 1)

                a = torch.flatten(a, 1)
                v = torch.flatten(v, 1)



                v_std_in = v_std.view(B, -1, C, H, W)
                v_std_in = v_std_in.permute(0, 2, 1, 3, 4)

                a_std_in = F.adaptive_avg_pool2d(a_std, 1)
                v_std_in = F.adaptive_avg_pool3d(v_std_in, 1)

                a_std_in = torch.flatten(a_std_in, 1)
                v_std_in = torch.flatten(v_std_in, 1)



                a_std_fc=self.audio_variance_estimator(a_std_in.detach())
                v_std_fc=self.visual_variance_estimator(v_std_in.detach())

                a_std_fc = (a_std_fc * 0.5).exp()
                v_std_fc = (v_std_fc * 0.5).exp()

                if self.training:
                    if self.args.current_epoch<self.args.cylcle_epoch+10:
                        target_weight_a,target_weight_v=1,1
                    else:
                        target_weight_a,target_weight_v=2*v_std_fc**2/(v_std_fc**2+a_std_fc**2),2*a_std_fc**2/(v_std_fc**2+a_std_fc**2)

                        # teaching force
                        # a_varinace_label=torch.unsqueeze(a_varinace_label.float(),dim=1).cuda()
                        # v_varinace_label=torch.unsqueeze(v_varinace_label.float(),dim=1).cuda()
                        # weight_a,weight_v=2*v_varinace_label**2/(v_varinace_label**2+a_varinace_label**2),2*a_varinace_label**2/(v_varinace_label**2+a_varinace_label**2)
                else:
                    target_weight_a,target_weight_v=2*v_std_fc**2/(v_std_fc**2+a_std_fc**2),2*a_std_fc**2/(v_std_fc**2+a_std_fc**2)


                # print(target_weight_a.shape)
                weight_a=target_weight_a/self.args.audio_depend
                weight_v=target_weight_v/self.args.visual_depend

                weight_a,weight_v=2*weight_a/(weight_a+weight_v),2*weight_v/(weight_a+weight_v)
                
                # print(weight_a,weight_v)


                a_out, v_out, out = self.fusion_module(a+weight_a*a, v+weight_v*v)  # av 是原来的，out是融合结果

                return a, v, out, a_feature, v_feature, a_mul, a_std, v_mul, v_std, a_out, v_out,a_std_fc,v_std_fc,weight_a,weight_v
            else:
                a = self.audio_net(audio)  # only feature
                v = self.visual_net(visual)

                a_feature = a
                v_feature = v

                (_, C, H, W) = v.size()
                B = a.size()[0]
                v = v.view(B, -1, C, H, W)
                v = v.permute(0, 2, 1, 3, 4)

                a = F.adaptive_avg_pool2d(a, 1)
                v = F.adaptive_avg_pool3d(v, 1)

                a = torch.flatten(a, 1)
                v = torch.flatten(v, 1)



                # v_std_in = v_std.view(B, -1, C, H, W)
                # v_std_in = v_std_in.permute(0, 2, 1, 3, 4)

                # a_std_in = F.adaptive_avg_pool2d(a_std, 1)
                # v_std_in = F.adaptive_avg_pool3d(v_std_in, 1)

                # a_std_in = torch.flatten(a_std_in, 1)
                # v_std_in = torch.flatten(v_std_in, 1)



                a_std_fc=self.audio_variance_estimator(a.detach())
                v_std_fc=self.visual_variance_estimator(v.detach())

                a_std_fc = (a_std_fc * 0.5).exp()
                v_std_fc = (v_std_fc * 0.5).exp()

                if self.training:
                    if self.args.current_epoch<self.args.cylcle_epoch+10:
                        weight_a,weight_v=1,1
                    else:
                        weight_a,weight_v=2*v_std_fc**2/(v_std_fc**2+a_std_fc**2),2*a_std_fc**2/(v_std_fc**2+a_std_fc**2)

                        # teaching force
                        # a_varinace_label=torch.unsqueeze(a_varinace_label.float(),dim=1).cuda()
                        # v_varinace_label=torch.unsqueeze(v_varinace_label.float(),dim=1).cuda()
                        # weight_a,weight_v=2*v_varinace_label**2/(v_varinace_label**2+a_varinace_label**2),2*a_varinace_label**2/(v_varinace_label**2+a_varinace_label**2)
                else:
                    weight_a,weight_v=2*v_std_fc**2/(v_std_fc**2+a_std_fc**2),2*a_std_fc**2/(v_std_fc**2+a_std_fc**2)
                # # a_out=self.unimodal_fc(a)
                # # v_out=self.unimodal_fc(v)
                # weight_a,weight_v=1,1
                # print(weight_a,weight_v)
                a_out, v_out, out = self.fusion_module(a+weight_a*a, v+weight_v*v)  # av 是原来的，out是融合结果

                return a, v, out, a_feature, v_feature, 0, 0, 0, 0, a_out, v_out,a_std_fc,v_std_fc

                # return a, v, out, a_feature, v_feature, 0, 0, 0, 0, a, v
        elif self.modality == 'visual':

            if self.pe:
                v, v_mul, v_std = self.visual_net(visual)

                v_feature = v

                (_, C, H, W) = v.size()
                B = self.args.batch_size
                v = v.view(B, -1, C, H, W)
                v = v.permute(0, 2, 1, 3, 4)

                v = F.adaptive_avg_pool3d(v, 1)

                v = torch.flatten(v, 1)

                out = self.visual_classifier(v)

                a = torch.zeros_like(v)

                return a, v, out, v_feature, v_feature, 0, 0, v_mul, v_std


            else:

                v = self.visual_net(visual)

                v_feature = v

                (_, C, H, W) = v.size()
                B = self.args.batch_size
                v = v.view(B, -1, C, H, W)

                v = v.permute(0, 2, 1, 3, 4)

                v = F.adaptive_avg_pool3d(v, 1)

                v = torch.flatten(v, 1)

                out = self.visual_classifier(v)

                a = torch.zeros_like(v)

                # print(11111111111111)

                return a, v, out, v_feature, v_feature, 0, 0, 0, 0

        elif self.modality == 'audio':

            if self.pe:
                a, a_mul, a_std = self.audio_net(audio)  # only feature

                a_feature = a
                a = F.adaptive_avg_pool2d(a, 1)

                a = torch.flatten(a, 1)
                out = self.audio_classifier(a)
                v = torch.zeros_like(a)

                return a, v, out, a_feature, a_feature, a_mul, a_std, 0, 0

            else:

                a = self.audio_net(audio)  # only feature
                a_feature = a

                a = F.adaptive_avg_pool2d(a, 1)

                a = torch.flatten(a, 1)

                out = self.audio_classifier(a)
                v = torch.zeros_like(a)

                return a, v, out, a_feature, a_feature, 0, 0, 0, 0
        else:
            return 0, 0, 0
   



class AVClassifier_SiMM(nn.Module):
    def __init__(self, args):
        super(AVClassifier_SiMM, self).__init__()

        fusion = args.fusion_method
        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 34
        elif args.dataset == 'kinect400':
            n_classes = 400
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        if fusion == 'sum':
            self.fusion_module = SumFusion_AUXI(output_dim=n_classes)
        elif fusion == 'concat':
            if args.dataset == 'kinect400':
                self.fusion_module = ConcatFusion_AUXI(output_dim=n_classes, input_dim=1024)
            else:
                self.fusion_module = ConcatFusion_AUXI(output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM_AUXI(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion_AUXI(output_dim=n_classes, x_gate=True)
        elif fusion == 'share':
            self.fusion_module = ShareWeightFusion_AUXI(output_dim=n_classes)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        if args.modality == 'full':
            self.audio_net = resnet18(modality='audio', args=args)
            self.visual_net = resnet18(modality='visual', args=args)

        if args.modality == 'visual':
            if args.dataset == 'kinect400':
                print("resnet50")
                self.visual_net = resnet18(modality='visual', args=args)
                self.visual_classifier = nn.Linear(512, n_classes)
            else:
                self.visual_net = resnet18(modality='visual', args=args)
                self.visual_classifier = nn.Linear(512, n_classes)
        if args.modality == 'audio':
            if args.dataset == 'kinect400':
                print("resnet50")
                self.audio_net = resnet18(modality='audio', args=args)
                self.audio_classifier = nn.Linear(512, n_classes)
            else:
                self.audio_net = resnet18(modality='audio', args=args)
                self.audio_classifier = nn.Linear(512, n_classes)
        self.pe = args.pe
        self.modality = args.modality
        self.args = args

        self.audio_variance_estimator=nn.Sequential(nn.Linear(1024,512),nn.Dropout(0.1),nn.Linear(512,2))


    def forward(self, audio, visual):

        if self.modality == 'full':

            if self.pe:
                a, a_mul, a_std = self.audio_net(audio)  # only feature
                v, v_mul, v_std = self.visual_net(visual)

                if self.args.drop:
                    a, v, p = modality_drop(a, v, self.args.p, args=self.args)

                a_feature = a
                v_feature = v
                # print(p.shape)
                p_index=p.view(p.shape[0],p.shape[1])
                (_, C, H, W) = v.size()
                B = a.size()[0]
                v = v.view(B, -1, C, H, W)
                v = v.permute(0, 2, 1, 3, 4)

                a = F.adaptive_avg_pool2d(a, 1)
                v = F.adaptive_avg_pool3d(v, 1)

                a = torch.flatten(a, 1)
                v = torch.flatten(v, 1)

                a_v_concat=torch.cat([a,v],dim=1)

                a_std_fc=self.audio_variance_estimator(a_v_concat.detach())

                weight=F.softmax(a_std_fc,dim=1)
                weight_p=2*weight*p_index.float()
                weight=weight_p
                if self.training:
                    if self.args.current_epoch<self.args.cylcle_epoch+10:
                        weight_a,weight_v=1,1
                    else:
                        weight_a,weight_v=weight[:,0],weight[:,1]
                        weight_a=torch.unsqueeze(weight_a,dim=1)
                        weight_v=torch.unsqueeze(weight_v,dim=1)
                else:
                    weight_a,weight_v=weight[:,0],weight[:,1]
                    weight_a=torch.unsqueeze(weight_a,dim=1)
                    weight_v=torch.unsqueeze(weight_v,dim=1)                   

                a_out, v_out, out = self.fusion_module(weight_a*a, weight_v*v)  # av 是原来的，out是融合结果

                return a, v, out, a_feature, v_feature, a_mul, a_std, v_mul, v_std, a_out, v_out
            else:
                a = self.audio_net(audio)  # only feature
                v = self.visual_net(visual)

                if self.args.drop:
                    a, v, p = modality_drop(a, v, self.args.p, args=self.args)

                a_feature = a
                v_feature = v
                # print(p.shape)
                p_index=p.view(p.shape[0],p.shape[1])

                (_, C, H, W) = v.size()
                B = a.size()[0]
                v = v.view(B, -1, C, H, W)
                v = v.permute(0, 2, 1, 3, 4)

                a = F.adaptive_avg_pool2d(a, 1)
                v = F.adaptive_avg_pool3d(v, 1)

                a = torch.flatten(a, 1)
                v = torch.flatten(v, 1)

                a_v_concat=torch.cat([a,v],dim=1)

                a_std_fc=self.audio_variance_estimator(a_v_concat.detach())

                weight=F.softmax(a_std_fc,dim=1)
                weight_p=2*weight*p_index.float()
                weight=weight_p
                # print(weight.shape)
                if self.training:
                    if self.args.current_epoch<self.args.cylcle_epoch+10:
                        weight_a,weight_v=1,1
                    else:
                        weight_a,weight_v=weight[:,0],weight[:,1]
                        weight_a=torch.unsqueeze(weight_a,dim=1)
                        weight_v=torch.unsqueeze(weight_v,dim=1)
                else:
                    if self.args.current_epoch<self.args.cylcle_epoch+10:
                        weight_a,weight_v=1,1
                    else:
                        weight_a,weight_v=weight[:,0],weight[:,1]
                        weight_a=torch.unsqueeze(weight_a,dim=1)
                        weight_v=torch.unsqueeze(weight_v,dim=1)           

                    # print(weight_a.shape,a.shape)
                a_out, v_out, out = self.fusion_module(weight_a*a, weight_v*v)  # av 是原来的，out是融合结果

                # a, v, out = self.fusion_module(a, v)  # av 是原来的，out是融合结果

                return a, v, out, a_feature, v_feature, 0, 0, 0, 0, a_out, v_out
        elif self.modality == 'visual':

            if self.pe:
                v, v_mul, v_std = self.visual_net(visual)

                v_feature = v

                (_, C, H, W) = v.size()
                B = self.args.batch_size
                v = v.view(B, -1, C, H, W)
                v = v.permute(0, 2, 1, 3, 4)

                v = F.adaptive_avg_pool3d(v, 1)

                v = torch.flatten(v, 1)

                out = self.visual_classifier(v)

                a = torch.zeros_like(v)

                return a, v, out, v_feature, v_feature, 0, 0, v_mul, v_std


            else:

                v = self.visual_net(visual)

                v_feature = v

                (_, C, H, W) = v.size()
                B = self.args.batch_size
                v = v.view(B, -1, C, H, W)

                v = v.permute(0, 2, 1, 3, 4)

                v = F.adaptive_avg_pool3d(v, 1)

                v = torch.flatten(v, 1)

                out = self.visual_classifier(v)

                a = torch.zeros_like(v)

                # print(11111111111111)

                return a, v, out, v_feature, v_feature, 0, 0, 0, 0

        elif self.modality == 'audio':

            if self.pe:
                a, a_mul, a_std = self.audio_net(audio)  # only feature

                a_feature = a
                a = F.adaptive_avg_pool2d(a, 1)

                a = torch.flatten(a, 1)
                out = self.audio_classifier(a)
                v = torch.zeros_like(a)

                return a, v, out, a_feature, a_feature, a_mul, a_std, 0, 0

            else:

                a = self.audio_net(audio)  # only feature
                a_feature = a

                a = F.adaptive_avg_pool2d(a, 1)

                a = torch.flatten(a, 1)

                out = self.audio_classifier(a)
                v = torch.zeros_like(a)

                return a, v, out, a_feature, a_feature, 0, 0, 0, 0
        else:
            return 0, 0, 0




class AVClassifier_AUXI_Weight(nn.Module):
    def __init__(self, args):
        super(AVClassifier_AUXI_Weight, self).__init__()

        fusion = args.fusion_method
        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 34
        elif args.dataset == 'kinect400':
            n_classes = 400
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        if fusion == 'sum':
            self.fusion_module = SumFusion_AUXI(output_dim=n_classes)
        elif fusion == 'concat':
            if args.dataset == 'kinect400':
                self.fusion_module = ConcatFusion_AUXI(output_dim=n_classes, input_dim=1024)
            else:
                self.fusion_module = ConcatFusion_AUXI(output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM_AUXI(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion_AUXI(output_dim=n_classes, x_gate=True)
        elif fusion == 'share':
            self.fusion_module = ShareWeightFusion_AUXI(output_dim=n_classes)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        if args.modality == 'full':
            self.audio_net = resnet18_weight(modality='audio', args=args)
            self.visual_net = resnet18_weight(modality='visual', args=args)
            # self.visual_net.mu_dul_backbone = self.audio_net.mu_dul_backbone
            # self.visual_net.logvar_dul_backbone = self.audio_net.logvar_dul_backbone

        if args.modality == 'visual':
            if args.dataset == 'kinect400':
                print("resnet50")
                self.visual_net = resnet18_weight(modality='visual', args=args)
                self.visual_classifier = nn.Linear(512, n_classes)
            else:
                self.visual_net = resnet18_weight(modality='visual', args=args)
                self.visual_classifier = nn.Linear(512, n_classes)
        if args.modality == 'audio':
            if args.dataset == 'kinect400':
                print("resnet50")
                self.audio_net = resnet18_weight(modality='audio', args=args)
                self.audio_classifier = nn.Linear(512, n_classes)
            else:
                self.audio_net = resnet18_weight(modality='audio', args=args)
                self.audio_classifier = nn.Linear(512, n_classes)
        self.pe = args.pe
        self.modality = args.modality
        self.args = args


        self.visual_variance_estimator=nn.Sequential(nn.Linear(512,256),nn.Dropout(0.1),nn.Linear(256,1))
        self.audio_variance_estimator=nn.Sequential(nn.Linear(512,256),nn.Dropout(0.1),nn.Linear(256,1))

    def forward(self, audio, visual,labels):

        if self.modality == 'full':
            # if visual.shape[1]==1:
            #     visual=torch.squeeze(visual,dim=1)
            if self.pe:
                a, a_mul, a_std,a_centers = self.audio_net(audio,labels)  # only feature
                v, v_mul, v_std,v_centers = self.visual_net(visual,labels)
                # print(bool(self.args.drop))
                if self.args.drop:
                    a, v, p = modality_drop(a, v, self.args.p, args=self.args)

                a_feature = a
                v_feature = v

                (_, C, H, W) = v.size()
                B = a.size()[0]
                v = v.view(B, -1, C, H, W)
                v = v.permute(0, 2, 1, 3, 4)

                a = F.adaptive_avg_pool2d(a, 1)
                v = F.adaptive_avg_pool3d(v, 1)


                a = torch.flatten(a, 1)
                v = torch.flatten(v, 1)

                # print(a.shape,v.shape)

                a_std_fc=self.audio_variance_estimator(a.detach())
                v_std_fc=self.visual_variance_estimator(v.detach())

                a_std_fc = (a_std_fc * 0.5).exp()
                v_std_fc = (v_std_fc * 0.5).exp()

                if self.training:
                    if self.args.current_epoch<self.args.cylcle_epoch+10:
                    # if self.args.current_epoch<10:
                        weight_a,weight_v=1,1
                    else:
                        weight_a,weight_v=2*v_std_fc**2/(v_std_fc**2+a_std_fc**2),2*a_std_fc**2/(v_std_fc**2+a_std_fc**2)

                else:
                    weight_a,weight_v=2*v_std_fc**2/(v_std_fc**2+a_std_fc**2),2*a_std_fc**2/(v_std_fc**2+a_std_fc**2)

                a_out, v_out, out = self.fusion_module(a+weight_a*a, v+weight_v*v)  # av 是原来的，out是融合结果

                return a, v, out, a_feature, v_feature, a_mul, a_std, v_mul, v_std, a_out, v_out,a_std_fc,v_std_fc,a_centers,v_centers
            else:
                a = self.audio_net(audio)  # only feature
                v = self.visual_net(visual)

                a_feature = a
                v_feature = v

                (_, C, H, W) = v.size()
                B = a.size()[0]
                v = v.view(B, -1, C, H, W)
                v = v.permute(0, 2, 1, 3, 4)

                a = F.adaptive_avg_pool2d(a, 1)
                v = F.adaptive_avg_pool3d(v, 1)

                a = torch.flatten(a, 1)
                v = torch.flatten(v, 1)

                # a_out=self.unimodal_fc(a)
                # v_out=self.unimodal_fc(v)

                a_out, v_out, out = self.fusion_module(a, v)  # av 是原来的，out是融合结果

                return a, v, out, a_feature, v_feature, 0, 0, 0, 0, a_out, v_out
        elif self.modality == 'visual':

            if self.pe:
                v, v_mul, v_std = self.visual_net(visual)

                v_feature = v

                (_, C, H, W) = v.size()
                B = self.args.batch_size
                v = v.view(B, -1, C, H, W)
                v = v.permute(0, 2, 1, 3, 4)

                v = F.adaptive_avg_pool3d(v, 1)

                v = torch.flatten(v, 1)

                out = self.visual_classifier(v)

                a = torch.zeros_like(v)

                return a, v, out, v_feature, v_feature, 0, 0, v_mul, v_std


            else:

                v = self.visual_net(visual)

                v_feature = v

                (_, C, H, W) = v.size()
                B = self.args.batch_size
                v = v.view(B, -1, C, H, W)

                v = v.permute(0, 2, 1, 3, 4)

                v = F.adaptive_avg_pool3d(v, 1)

                v = torch.flatten(v, 1)

                out = self.visual_classifier(v)

                a = torch.zeros_like(v)

                # print(11111111111111)

                return a, v, out, v_feature, v_feature, 0, 0, 0, 0

        elif self.modality == 'audio':

            if self.pe:
                a, a_mul, a_std = self.audio_net(audio)  # only feature

                a_feature = a
                a = F.adaptive_avg_pool2d(a, 1)

                a = torch.flatten(a, 1)
                out = self.audio_classifier(a)
                v = torch.zeros_like(a)

                return a, v, out, a_feature, a_feature, a_mul, a_std, 0, 0

            else:

                a = self.audio_net(audio)  # only feature
                a_feature = a

                a = F.adaptive_avg_pool2d(a, 1)

                a = torch.flatten(a, 1)

                out = self.audio_classifier(a)
                v = torch.zeros_like(a)

                return a, v, out, a_feature, a_feature, 0, 0, 0, 0
        else:
            return 0, 0, 0





class AVClassifier_AUXI_eau(nn.Module):
    def __init__(self, args):
        super(AVClassifier_AUXI_eau, self).__init__()

        fusion = args.fusion_method
        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 34
        elif args.dataset == 'kinect400':
            n_classes = 400
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        if fusion == 'sum':
            self.fusion_module = SumFusion_AUXI(output_dim=n_classes)
        elif fusion == 'concat':
            if args.dataset == 'kinect400':
                self.fusion_module = ConcatFusion_AUXI(output_dim=n_classes, input_dim=1024)
            else:
                self.fusion_module = ConcatFusion_AUXI(output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM_AUXI(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion_AUXI(output_dim=n_classes, x_gate=True)
        elif fusion == 'share':
            self.fusion_module = ShareWeightFusion_AUXI(output_dim=n_classes)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        if args.modality == 'full':
            self.audio_net = resnet18_weight(modality='audio', args=args)
            self.visual_net = resnet18_weight(modality='visual', args=args)
            # self.visual_net.mu_dul_backbone = self.audio_net.mu_dul_backbone
            # self.visual_net.logvar_dul_backbone = self.audio_net.logvar_dul_backbone

        if args.modality == 'visual':
            if args.dataset == 'kinect400':
                print("resnet50")
                self.visual_net = resnet18_weight(modality='visual', args=args)
                self.visual_classifier = nn.Linear(512, n_classes)
            else:
                self.visual_net = resnet18_weight(modality='visual', args=args)
                self.visual_classifier = nn.Linear(512, n_classes)
        if args.modality == 'audio':
            if args.dataset == 'kinect400':
                print("resnet50")
                self.audio_net = resnet18_weight(modality='audio', args=args)
                self.audio_classifier = nn.Linear(512, n_classes)
            else:
                self.audio_net = resnet18_weight(modality='audio', args=args)
                self.audio_classifier = nn.Linear(512, n_classes)
        self.pe = args.pe
        self.modality = args.modality
        self.args = args


        self.visual_variance_estimator=nn.Sequential(nn.Linear(512,256),nn.Dropout(0.1),nn.Linear(256,1))
        self.audio_variance_estimator=nn.Sequential(nn.Linear(512,256),nn.Dropout(0.1),nn.Linear(256,1))

    def forward(self, audio, visual,labels):

        if self.modality == 'full':
            # if visual.shape[1]==1:
            #     visual=torch.squeeze(visual,dim=1)
            if self.pe:
                a, a_mul, a_std,a_centers = self.audio_net(audio,labels)  # only feature
                v, v_mul, v_std,v_centers = self.visual_net(visual,labels)
                # print(bool(self.args.drop))
                if self.args.drop:
                    a, v, p = modality_drop(a, v, self.args.p, args=self.args)

                a_feature = a
                v_feature = v

                (_, C, H, W) = v.size()
                B = a.size()[0]
                v = v.view(B, -1, C, H, W)
                v = v.permute(0, 2, 1, 3, 4)

                a = F.adaptive_avg_pool2d(a, 1)
                v = F.adaptive_avg_pool3d(v, 1)


                a = torch.flatten(a, 1)
                v = torch.flatten(v, 1)

                # print(a.shape,v.shape)

                a_std_fc=F.adaptive_avg_pool2d(a_std, 1)
                a_std_fc = torch.flatten(a_std_fc, 1)

                v_std_in = v_std.view(B, -1, C, H, W)
                v_std_in = v_std_in.permute(0, 2, 1, 3, 4) 
                v_std_fc=F.adaptive_avg_pool3d(v_std_in, 1)
                v_std_fc = torch.flatten(v_std_fc, 1)

                a_std_fc = (a_std_fc * 0.5).exp()
                v_std_fc = (v_std_fc * 0.5).exp()

    
                weight_a,weight_v=2*v_std_fc**2/(v_std_fc**2+a_std_fc**2),2*a_std_fc**2/(v_std_fc**2+a_std_fc**2)

                a_out, v_out, out = self.fusion_module(weight_a*a, weight_v*v)  # av 是原来的，out是融合结果

                return a, v, out, a_feature, v_feature, a_mul, a_std, v_mul, v_std, a_out, v_out,a_std_fc,v_std_fc,a_centers,v_centers
            else:
                a = self.audio_net(audio)  # only feature
                v = self.visual_net(visual)

                a_feature = a
                v_feature = v

                (_, C, H, W) = v.size()
                B = a.size()[0]
                v = v.view(B, -1, C, H, W)
                v = v.permute(0, 2, 1, 3, 4)

                a = F.adaptive_avg_pool2d(a, 1)
                v = F.adaptive_avg_pool3d(v, 1)

                a = torch.flatten(a, 1)
                v = torch.flatten(v, 1)

                # a_out=self.unimodal_fc(a)
                # v_out=self.unimodal_fc(v)

                a_out, v_out, out = self.fusion_module(a, v)  # av 是原来的，out是融合结果

                return a, v, out, a_feature, v_feature, 0, 0, 0, 0, a_out, v_out
        elif self.modality == 'visual':

            if self.pe:
                v, v_mul, v_std = self.visual_net(visual)

                v_feature = v

                (_, C, H, W) = v.size()
                B = self.args.batch_size
                v = v.view(B, -1, C, H, W)
                v = v.permute(0, 2, 1, 3, 4)

                v = F.adaptive_avg_pool3d(v, 1)

                v = torch.flatten(v, 1)

                out = self.visual_classifier(v)

                a = torch.zeros_like(v)

                return a, v, out, v_feature, v_feature, 0, 0, v_mul, v_std


            else:

                v = self.visual_net(visual)

                v_feature = v

                (_, C, H, W) = v.size()
                B = self.args.batch_size
                v = v.view(B, -1, C, H, W)

                v = v.permute(0, 2, 1, 3, 4)

                v = F.adaptive_avg_pool3d(v, 1)

                v = torch.flatten(v, 1)

                out = self.visual_classifier(v)

                a = torch.zeros_like(v)

                # print(11111111111111)

                return a, v, out, v_feature, v_feature, 0, 0, 0, 0

        elif self.modality == 'audio':

            if self.pe:
                a, a_mul, a_std = self.audio_net(audio)  # only feature

                a_feature = a
                a = F.adaptive_avg_pool2d(a, 1)

                a = torch.flatten(a, 1)
                out = self.audio_classifier(a)
                v = torch.zeros_like(a)

                return a, v, out, a_feature, a_feature, a_mul, a_std, 0, 0

            else:

                a = self.audio_net(audio)  # only feature
                a_feature = a

                a = F.adaptive_avg_pool2d(a, 1)

                a = torch.flatten(a, 1)

                out = self.audio_classifier(a)
                v = torch.zeros_like(a)

                return a, v, out, a_feature, a_feature, 0, 0, 0, 0
        else:
            return 0, 0, 0




class AVClassifier_SWIN(nn.Module):
    def __init__(self, args):
        super(AVClassifier_SWIN, self).__init__()
        from models.swin_transformer import SwinTransformer

        fusion = args.fusion_method
        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 34
        elif args.dataset == 'kinect400':
            n_classes = 400
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        if fusion == 'sum':
            self.fusion_module = SumFusion_AUXI(output_dim=n_classes, input_dim=2048)
        elif fusion == 'concat':
            if args.dataset == 'kinect400':
                self.fusion_module = ConcatFusion_AUXI(output_dim=n_classes, input_dim=1024)
            else:
                self.fusion_module = ConcatFusion_AUXI(output_dim=n_classes, input_dim=2048)
        elif fusion == 'film':
            self.fusion_module = FiLM_AUXI(output_dim=n_classes, x_film=True, input_dim=2048)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion_AUXI(output_dim=n_classes, x_gate=True, input_dim=1024)
        elif fusion == 'share':
            self.fusion_module = ShareWeightFusion_AUXI(output_dim=n_classes, input_dim=2048)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        if args.modality == 'full':
            self.audio_net = SwinTransformer(args=args, modality='audio', num_classes=n_classes, in_chans=3)
            self.visual_net = SwinTransformer(args=args, modality='visual', num_classes=n_classes)

            if args.pretrain:
                print("using pretrain")
                self.audio_net.load_state_dict(torch.load("swin_base_patch4_window7_224_22k.pth")['model'],
                                               strict=False)
                self.visual_net.load_state_dict(torch.load("swin_base_patch4_window7_224_22k.pth")['model'],
                                                strict=False)

        if args.modality == 'visual':
            self.visual_net = SwinTransformer(modality='visual', num_classes=n_classes)
            if args.pretrain:
                print("using pretrain")
                self.visual_net.load_state_dict(torch.load("swin_tiny_patch4_window7_224.pth")['model'], strict=False)
            self.visual_classifier = nn.Linear(768, n_classes)
        if args.modality == 'audio':
            self.audio_net = SwinTransformer(modality='audio', in_chans=1, num_classes=n_classes, img_size=224)
            if args.pretrain:
                print("using pretrain")
                self.audio_net.load_state_dict(torch.load("ckpt_epoch_47.pth")['model'], strict=False)
            self.audio_classifier = nn.Linear(768, n_classes)
        self.pe = args.pe
        self.modality = args.modality
        self.args = args
        self.pool1d = nn.AdaptiveAvgPool1d(1)
        # self.audio_mu = nn.Linear(512, 512)
        # self.audio_logval = PCME(512, 512, 256)
        # self.visual_mu = nn.Linear(512, 512)
        # self.visual_logval = PCME(512, 512, 256)

    def forward(self, audio, visual):
        audio = torch.repeat_interleave(audio, 3, 1)

        if self.modality == 'full':

            if self.pe:
                a, a_mul, a_std = self.audio_net(audio)  # only feature
                v, v_mul, v_std = self.visual_net(visual)

                a_feature = a
                v_feature = v

                (_, C, H, W) = v.size()
                B = a.size()[0]
                v = v.view(B, -1, C, H, W)
                v = v.permute(0, 2, 1, 3, 4)

                a = F.adaptive_avg_pool2d(a, 1)
                v = F.adaptive_avg_pool3d(v, 1)

                a = torch.flatten(a, 1)
                v = torch.flatten(v, 1)

                # v=v*0

                a, v, out = self.fusion_module(a, v)  # av 是原来的，out是融合结果

                return a, v, out, a_feature, v_feature, a_mul, a_std, v_mul, v_std
            else:
                # audio = torch.repeat_interleave(audio, 3, 1)

                a = self.audio_net(audio)  # only feature
                v = self.visual_net(visual)

                a_feature = a
                v_feature = v

                a = torch.flatten(a, 1)
                v = torch.flatten(v, 1)

                out_a, out_v, out = self.fusion_module(a, v)  # av 是原来的，out是融合结果

                return a, v, out, a_feature, v_feature, torch.zeros_like(a), torch.zeros_like(a), torch.zeros_like(
                    a), torch.zeros_like(a), out_a, out_v
        elif self.modality == 'visual':

            if self.pe:
                v, v_mul, v_std = self.visual_net(visual)

                v_feature = v

                (_, C, H, W) = v.size()
                B = self.args.batch_size
                v = v.view(B, -1, C, H, W)
                v = v.permute(0, 2, 1, 3, 4)

                v = F.adaptive_avg_pool3d(v, 1)

                v = torch.flatten(v, 1)

                out = self.visual_classifier(v)

                a = torch.zeros_like(v)

                return a, v, out, v_feature, v_feature, 0, 0, v_mul, v_std


            else:

                v = self.visual_net(visual)

                v_feature = v
                # print(v.shape)

                (_, C, L) = v.size()
                B = self.args.batch_size
                v = v.view(B, -1, C, L)

                v = v.permute(0, 2, 1, 3)

                v = F.adaptive_avg_pool2d(v, 1)
                # print(v.shape)

                v = torch.flatten(v, 1)

                out = self.visual_classifier(v)

                a = torch.zeros_like(v)

                return a, v, out, v_feature, v_feature, 0, 0, 0, 0

        elif self.modality == 'audio':

            if self.pe:
                a, a_mul, a_std = self.audio_net(audio)  # only feature

                a_feature = a
                a = self.pool1d(a)

                a = torch.flatten(a, 1)
                out = self.audio_classifier(a)
                v = torch.zeros_like(a)

                return a, v, out, a_feature, a_feature, a_mul, a_std, 0, 0

            else:
                # print(audio.shape)
                # audio = (audio - (-2.8829)) / 3.01
                # audio = (audio - (-6.8442)) /5.1567
                # print(audio.mean(),audio.std())

                # audio = torch.repeat_interleave(audio, 3, 1)
                a = self.audio_net(audio)  # only feature
                a_feature = a
                # print(a.shape)

                a = self.pool1d(a)

                a = torch.flatten(a, 1)

                out = self.audio_classifier(a)
                v = torch.zeros_like(a)

                return a, v, out, a_feature, a_feature, 0, 0, 0, 0
        else:
            return 0, 0, 0


class AVClassifier_Visaul(nn.Module):
    def __init__(self, args):
        super(AVClassifier_Visaul, self).__init__()

        fusion = args.fusion_method
        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 31
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        if fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            self.fusion_module = ConcatFusion(output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion(output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        self.visual_net = resnet18(modality='visual', args=args)
        self.pe = args.pe
        self.fc = nn.Linear(512, n_classes)

    def forward(self, audio, visual):
        if self.pe:
            v, v_mul, v_std = self.visual_net(visual)

            v_feature = v

            (_, C, H, W) = v.size()
            B = v.size()[0]
            v = v.view(B, -1, C, H, W)
            v = v.permute(0, 2, 1, 3, 4)

            v = F.adaptive_avg_pool3d(v, 1)

            v = torch.flatten(v, 1)

            # v=v*0

            out = self.fc(v)

            return 0, v, out, 0, v_feature, 0, 0, v_mul, v_std
        else:
            v = self.visual_net(visual)

            v_feature = v

            (_, C, H, W) = v.size()
            B = v.size()[0]
            v = v.view(B, -1, C, H, W)
            v = v.permute(0, 2, 1, 3, 4)

            v = F.adaptive_avg_pool3d(v, 1)

            v = torch.flatten(v, 1)
            # v = v * 0
            out = self.fc(v)

            return 0, v, out, 0, v_feature, 0, 0, 0, 0


class AVClassifier_Audio(nn.Module):
    def __init__(self, args):
        super(AVClassifier_Audio, self).__init__()

        fusion = args.fusion_method
        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 34
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        if fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            self.fusion_module = ConcatFusion(output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion(output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        self.audio_net = resnet18(modality='audio', args=args)
        self.visual_net = resnet18(modality='visual', args=args)
        self.pe = args.pe
        self.fc = nn.Linear(512, n_classes)

    def forward(self, audio, visual):
        if self.pe:
            a, a_mul, a_std = self.audio_net(audio)  # only feature
            v, v_mul, v_std = self.visual_net(visual)

            a_feature = a
            v_feature = v

            (_, C, H, W) = v.size()
            B = a.size()[0]
            v = v.view(B, -1, C, H, W)
            v = v.permute(0, 2, 1, 3, 4)

            a = F.adaptive_avg_pool2d(a, 1)
            v = F.adaptive_avg_pool3d(v, 1)

            a = torch.flatten(a, 1)
            v = torch.flatten(v, 1)

            # v=v*0

            out = self.fc(a)

            return a, v, out, a_feature, v_feature, a_mul, a_std, v_mul, v_std
        else:
            a = self.audio_net(audio)  # only feature
            v = self.visual_net(visual)

            a_feature = a
            v_feature = v

            (_, C, H, W) = v.size()
            B = a.size()[0]
            v = v.view(B, -1, C, H, W)
            v = v.permute(0, 2, 1, 3, 4)

            a = F.adaptive_avg_pool2d(a, 1)
            v = F.adaptive_avg_pool3d(v, 1)

            a = torch.flatten(a, 1)
            v = torch.flatten(v, 1)
            # v = v * 0
            out = self.fc(a)

            return a, v, out, a_feature, v_feature, 0, 0, 0, 0


class AVClassifier_Unimodal(nn.Module):
    def __init__(self, args):
        super(AVClassifier_Unimodal, self).__init__()

        fusion = args.fusion_method
        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 34
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        if fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            self.fusion_module = ConcatFusion(output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion(output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        self.audio_net = resnet18(modality='audio', args=args)
        self.visual_net = resnet18(modality='visual', args=args)
        self.pe = args.pe
        self.fc = nn.Linear(512, n_classes)

    def forward(self, audio, visual):

        if self.pe:

            a, a_mul, a_std = self.audio_net(audio)  # only feature
            v, v_mul, v_std = self.visual_net(visual)

            a_feature = a
            v_feature = v

            (_, C, H, W) = v.size()
            B = a.size()[0]
            v = v.view(B, -1, C, H, W)
            v = v.permute(0, 2, 1, 3, 4)

            a = F.adaptive_avg_pool2d(a, 1)
            v = F.adaptive_avg_pool3d(v, 1)

            a = torch.flatten(a, 1)
            v = torch.flatten(v, 1)

            out = self.fc(a)

            return a, v, a, a_feature, v_feature, a_mul, a_std, v_mul, v_std
        else:
            a = self.audio_net(audio)  # only feature
            v = self.visual_net(visual)

            a_feature = a
            v_feature = v

            (_, C, H, W) = v.size()
            B = a.size()[0]
            v = v.view(B, -1, C, H, W)
            v = v.permute(0, 2, 1, 3, 4)

            a = F.adaptive_avg_pool2d(a, 1)
            v = F.adaptive_avg_pool3d(v, 1)

            a = torch.flatten(a, 1)
            v = torch.flatten(v, 1)

            out = self.fc(a)

            return a, v, a, a_feature, v_feature, 0, 0, 0, 0


class AVClassifier_PE(nn.Module):
    def __init__(self, args):
        super(AVClassifier_PE, self).__init__()

        fusion = args.fusion_method
        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 34
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        if fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            self.fusion_module = ConcatFusion(output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion(output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        self.audio_net = resnet18(modality='audio', args=args)
        self.visual_net = resnet18(modality='visual', args=args)

    def forward(self, audio, visual):

        a = self.audio_net(audio)  # only feature
        v = self.visual_net(visual)

        (_, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)

        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)

        a, v, out = self.fusion_module(a, v)  # av 是原来的，out是融合结果

        return a, v, out
