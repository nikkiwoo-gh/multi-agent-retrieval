import torch
import torch.nn as nn
import torch.nn.init
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm  # clip_grad_norm_ for 0.4.0, clip_grad_norm for 0.3.1
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X

def xavier_init_fc(fc):
    """Xavier initialization for the fully connected layer
    """
    r = np.sqrt(6.) / np.sqrt(fc.in_features +
                             fc.out_features)
    fc.weight.data.uniform_(-r, r)
    fc.bias.data.fill_(0)

class MFC(nn.Module):
    """
    Multi Fully Connected Layers
    """
    def __init__(self, fc_layers, dropout, have_dp=True, have_bn=False, have_last_bn=False):
        super(MFC, self).__init__()
        # fc layers
        self.n_fc = len(fc_layers)
        if self.n_fc > 1:
            if self.n_fc > 1:
                self.fc1 = nn.Linear(fc_layers[0], fc_layers[1])

            # dropout
            self.have_dp = have_dp
            if self.have_dp:
                self.dropout = nn.Dropout(p=dropout)

            # batch normalization
            self.have_bn = have_bn
            self.have_last_bn = have_last_bn
            if self.have_bn:
                if self.n_fc == 2 and self.have_last_bn:
                    self.bn_1 = nn.BatchNorm1d(fc_layers[1])

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        if self.n_fc > 1:
            xavier_init_fc(self.fc1)

    def forward(self, inputs):

        if self.n_fc <= 1:
            features = inputs

        elif self.n_fc == 2:
            features = self.fc1(inputs)
            # batch noarmalization
            if self.have_bn and self.have_last_bn:
                features = self.bn_1(features)
            if self.have_dp:
                features = self.dropout(features)

        return features

class Text_one_layer_encoder(nn.Module):
    """
    Section 3.2. Text-side Multi-level Encoding
    """

    def __init__(self, opt):
        super(Text_one_layer_encoder, self).__init__()
        self.text_norm = opt.text_norm
        self.dropout = nn.Dropout(p=opt.dropout)
        self.with_textual_mapping = opt.with_textual_mapping
        # multi fc layers
        if self.with_textual_mapping:
            self.text_mapping = MFC(opt.text_mapping_layers, opt.dropout, have_bn=True, have_last_bn=True)


    def forward(self, text, *args):
        # Embed word ids to vectors
        features = text


        # mapping to common space
        if self.with_textual_mapping:
            features = self.text_mapping(features)
        if self.text_norm:
            features = l2norm(features)

        if np.sum(np.isnan(features.data.cpu().numpy())) > 0:
            print('features is nan')

        return features

class Video_encoder(nn.Module):
    """
    Section 3.1. Video-side Multi-level Encoding
    """

    def __init__(self, opt):
        super(Video_encoder, self).__init__()

        self.rnn_output_size = opt.visual_rnn_size * 2
        self.dropout = nn.Dropout(p=opt.dropout)
        self.visual_norm = opt.visual_norm
        self.concate = opt.vconcate

        # visual bidirectional rnn encoder
        self.rnn = nn.GRU(opt.visual_feat_dim, opt.visual_rnn_size, batch_first=True, bidirectional=True)

        # visual 1-d convolutional network
        self.convs1 = nn.ModuleList([
            nn.Conv2d(1, opt.visual_kernel_num, (window_size, self.rnn_output_size), padding=(window_size - 1, 0))
            for window_size in opt.visual_kernel_sizes
        ])

        # visual mapping
        self.visual_mapping = MFC(opt.visual_mapping_layers, opt.dropout, have_bn=True, have_last_bn=True)

    def forward(self, videos):
        """Extract video feature vectors."""

        videos, motions,videos_origin, lengths, vidoes_mask = videos

        # Level 1. Global Encoding by Mean Pooling According
        org_out = videos_origin

        # Level 2. Temporal-Aware Encoding by biGRU
        gru_init_out, _ = self.rnn(videos)
        mean_gru = Variable(torch.zeros(gru_init_out.size(0), self.rnn_output_size)).cuda()
        for i, batch in enumerate(gru_init_out):
            mean_gru[i] = torch.mean(batch[:lengths[i]], 0)
        gru_out = mean_gru
        gru_out = self.dropout(gru_out)

        # Level 3. Local-Enhanced Encoding by biGRU-CNN
        vidoes_mask = vidoes_mask.unsqueeze(2).expand(-1, -1, gru_init_out.size(2))  # (N,C,F1)
        gru_init_out = gru_init_out * vidoes_mask
        con_out = gru_init_out.unsqueeze(1)
        con_out = [F.relu(conv(con_out)).squeeze(3) for conv in self.convs1]
        con_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in con_out]
        con_out = torch.cat(con_out, 1)
        con_out = self.dropout(con_out)

        ##level 4 motion feature (e.g., slowfast)
        motion_out = motions
        # concatenation
        if self.concate == 'full':  # level 1+2+3
            features = torch.cat((gru_out, con_out, org_out,motion_out), 1)
        elif self.concate == 'reduced':  # level 2+3
            features = torch.cat((gru_out, con_out), 1)

        # mapping to common space
        features = self.visual_mapping(features)
        if self.visual_norm:
            features = l2norm(features)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(Video_encoder, self).load_state_dict(new_state)


class BaseModel(object):

    def state_dict(self):
        state_dict = [self.vid_encoder.state_dict(), self.text_encoder.state_dict() ,self.unify_decoder.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.vid_encoder.load_state_dict(state_dict[0])
        self.text_encoder.load_state_dict(state_dict[1])
        self.unify_decoder.load_state_dict(state_dict[2])
    
    def to(self, device):
        """Move model to specified device"""
        self.vid_encoder = self.vid_encoder.to(device)
        self.text_encoder = self.text_encoder.to(device)
        self.unify_decoder = self.unify_decoder.to(device)
        return self


class Improved_ITV(BaseModel):
    """
    Improved ITV network
    """
    def __init__(self, opt):
        # Build Models
        self.modelname = opt.postfix
        self.grad_clip = opt.grad_clip
        self.vid_encoder = Video_encoder(opt)
        self.text_encoder = Text_one_layer_encoder(opt)
        self.decoder_num_layer=len(opt.decoder_mapping_layers)

        if len(opt.decoder_mapping_layers)==2:
            self.unify_decoder =MFC(opt.decoder_mapping_layers, opt.dropout, have_bn=True, have_last_bn=True)
        elif len(opt.decoder_mapping_layers)==3:
            mapping_layer1 = [opt.decoder_mapping_layers[0],opt.decoder_mapping_layers[1]]
            mapping_layer2 = [opt.decoder_mapping_layers[1],opt.decoder_mapping_layers[2]]
            self.unify_decoder=nn.ModuleList([MFC(mapping_layer1, opt.dropout, have_bn=False, have_last_bn=False),
                                              MFC(mapping_layer2, opt.dropout, have_bn=True,
                                                  have_last_bn=True)])

        else:
            NotImplemented
        self.sigmod = nn.Sigmoid()

       
        self.unlikelihood=opt.unlikelihood
        self.ul_alpha = opt.ul_alpha
        self.concept_phrase = opt.concept_phrase




        params_end_text = list(self.text_encoder.parameters())
        params_end_vid = list(self.vid_encoder.parameters())
        params_unify_dec = list(self.unify_decoder.parameters())
        self.params_end_text = params_end_text
        self.params_end_vid = params_end_vid
        self.params_unify_dec=params_unify_dec
        params= params_end_text+params_end_vid+params_unify_dec
        self.params = params

    def embed_vis(self, vis_data, volatile=True,sigmoid_output=True):
        # video data
        frames, motions,mean_origin, video_lengths, vidoes_mask = vis_data

        if volatile:
            with torch.no_grad():
                frames = Variable(frames)
        else:
            frames = Variable(frames, requires_grad=True)
        if torch.cuda.is_available():
            frames = frames.cuda()

        if volatile:
            with torch.no_grad():
                motions = Variable(motions)
        else:
            motions = Variable(motions, requires_grad=True)

        if torch.cuda.is_available():
            motions = motions.cuda()

        if volatile:
            with torch.no_grad():
                mean_origin = Variable(mean_origin)
        else:
            mean_origin = Variable(mean_origin, requires_grad=True)


        if torch.cuda.is_available():
            mean_origin = mean_origin.cuda()

        if volatile:
            with torch.no_grad():
                vidoes_mask = Variable(vidoes_mask)
        else:
            vidoes_mask = Variable(vidoes_mask, requires_grad=True)

        if torch.cuda.is_available():
            vidoes_mask = vidoes_mask.cuda()
        vis_data = (frames, motions,mean_origin, video_lengths, vidoes_mask)
        embs = self.vid_encoder(vis_data)
        pred= self.vid_encoder(vis_data)
        if self.decoder_num_layer > 2:
            for decod in self.unify_decoder:
                pred = decod(pred)
        else:
            pred=self.unify_decoder(pred)
        sigmoid_out=self.sigmod(pred)

        if sigmoid_output:
            return embs,sigmoid_out
        else:
            return embs

    def embed_vis_emb_only(self, vis_data, volatile=True):
        # video data
        frames, motions,mean_origin, video_lengths, vidoes_mask = vis_data
        frames = Variable(frames, volatile=volatile)
        if torch.cuda.is_available():
            frames = frames.cuda()

        motions = Variable(motions, volatile=volatile)
        if torch.cuda.is_available():
            motions = motions.cuda()

        mean_origin = Variable(mean_origin, volatile=volatile)
        if torch.cuda.is_available():
            mean_origin = mean_origin.cuda()

        vidoes_mask = Variable(vidoes_mask, volatile=volatile)
        if torch.cuda.is_available():
            vidoes_mask = vidoes_mask.cuda()
        vis_data = (frames,motions, mean_origin, video_lengths, vidoes_mask)
        embs = self.vid_encoder(vis_data)
        return embs

    def embed_vis_concept_only(self, vis_data, volatile=True):
        # video data
        frames, motions,mean_origin, video_lengths, vidoes_mask = vis_data
        if volatile:
            with torch.no_grad():
                frames = Variable(frames)
        else:
            frames = Variable(frames, requires_grad=True)
        if torch.cuda.is_available():
            frames = frames.cuda()

        if volatile:
            with torch.no_grad():
                motions = Variable(motions)
        else:
            motions = Variable(motions, requires_grad=True)
        if torch.cuda.is_available():
            motions = motions.cuda()

        if volatile:
            with torch.no_grad():
                mean_origin = Variable(mean_origin)
        else:
            mean_origin = Variable(mean_origin, requires_grad=True)
        if torch.cuda.is_available():
            mean_origin = mean_origin.cuda()

        if volatile:
            with torch.no_grad():
                vidoes_mask = Variable(vidoes_mask)
        else:
            vidoes_mask = Variable(vidoes_mask, requires_grad=True)
        if torch.cuda.is_available():
            vidoes_mask = vidoes_mask.cuda()

        vis_data = (frames,motions, mean_origin, video_lengths, vidoes_mask)
        vid_embs = self.vid_encoder(vis_data)
        if self.decoder_num_layer > 2:
            for decod in self.unify_decoder:
                vid_embs = decod(vid_embs)
        else:
            vid_embs=self.unify_decoder(vid_embs)
        sigmoid_out=self.sigmod(vid_embs)
        return sigmoid_out


    def embed_txt(self, txt_data,  volatile=True,sigmoid_output=False):
        with torch.no_grad():
            text_emb = self.text_encoder(txt_data)
            if sigmoid_output:
                pred = self.text_encoder(txt_data)
                if self.decoder_num_layer > 2:
                    for decod in self.unify_decoder:
                        pred = decod(pred)
                else:
                    pred=self.unify_decoder(pred)  
                sigmoid_out=self.sigmod(pred)
                return text_emb, sigmoid_out
            else:
                return text_emb

    def embed_txt_concept_only(self, txt_data, volatile=True):
        # text data
        with torch.no_grad():
            text_emb = self.text_encoder(txt_data)
            if self.decoder_num_layer > 2:
                for decod in self.unify_decoder:
                    text_emb = decod(text_emb)
            else:
                text_emb=self.unify_decoder(text_emb)
            sigmoid_out=self.sigmod(text_emb)
            return sigmoid_out


