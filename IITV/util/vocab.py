import os
import pickle
from collections import Counter
import json
import argparse
import sys



ROOT_PATH = '/vireo00/nikki/AVS_data'
class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self, text_style):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.text_style = text_style

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx and 'bow' not in self.text_style:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)



class Concept_phase(object):
    """ concept wrapper"""

    def __init__(self):
        self.phrase2idx = {}
        self.idx2phrase = {}
        self.idx2contractIdx = {}
        self.phrase2contractphrase = {}
        self.idx2GlobalContractIdx = {}
        self.phrase2GlobalContractphrase = {}
        self.idx = 0
        self.num_contradict_paris = 0
        self.num_GlobalContradict_paris = 0

    def add_phrase(self, phrase):
        if phrase not in self.phrase2idx:
            self.phrase2idx[phrase] = self.idx
            self.idx2phrase[self.idx] = phrase
            self.idx += 1

    def add_contradict(self, contract_pair):
        contract_pair = contract_pair.strip().split('<->')
        ori_phrase_str = contract_pair[0]
        ori_phrase = ori_phrase_str.replace('_',' ')
        ori_phrase_idx = None
        if ori_phrase in self.phrase2idx:
            ori_phrase_idx = self.phrase2idx[ori_phrase]
        contract_phrase_str = contract_pair[1]
        contract_phrases = contract_phrase_str.split(',')
        contract_idxes = []
        icontra_ori_list= []
        for icontra in contract_phrases:
            icontra_ori = icontra.replace('_',' ')
            icontra_ori_list.append(icontra_ori)
            if icontra_ori in self.phrase2idx:
                idx = self.phrase2idx[icontra_ori]
                contract_idxes.append(idx)
        if (not ori_phrase_idx is None)&(len(contract_idxes)>0):
            self.idx2contractIdx[ori_phrase_idx] = contract_idxes
            self.phrase2contractphrase[ori_phrase] = ','.join(icontra_ori_list)
            self.num_contradict_paris += 1
        else:
            print('error in adding contradiction:%s'%ori_phrase)


    def add_global_contradict(self, contract_pair):
        contract_pair = contract_pair.strip().split('<->')
        ori_phrase_str = contract_pair[0]
        ori_phrase = ori_phrase_str.replace('_',' ')
        ori_phrase_idx = None
        if ori_phrase in self.phrase2idx:
            ori_phrase_idx = self.phrase2idx[ori_phrase]
        contract_phrase_str = contract_pair[1]
        contract_phrases = contract_phrase_str.split(',')
        contract_idxes = []
        icontra_ori_list= []
        for icontra in contract_phrases:
            icontra_ori = icontra.replace('_',' ')
            icontra_ori_list.append(icontra_ori)
            if icontra_ori in self.phrase2idx:
                idx = self.phrase2idx[icontra_ori]
                contract_idxes.append(idx)
        if (not ori_phrase_idx is None)&(len(contract_idxes)>0):
            self.idx2GlobalContractIdx[ori_phrase_idx] = contract_idxes
            self.phrase2GlobalContractphrase[ori_phrase] = ','.join(icontra_ori_list)
            self.num_GlobalContradict_paris += 1
        else:
            print('error in adding contradiction:%s'%ori_phrase)

    def __call__(self, phrase):
        if phrase not in self.phrase2idx:
            return self.phrase2idx['<unk>']
        return self.phrase2idx[phrase]

    def __len__(self):
        return len(self.phrase2idx)

class Concept_phrase(object):
    """ concept wrapper"""

    def __init__(self):
        self.phrase2idx = {}
        self.idx2phrase = {}
        self.idx2contractIdx = {}
        self.phrase2contractphrase = {}
        self.idx2GlobalContractIdx = {}
        self.phrase2GlobalContractphrase = {}
        self.idx = 0
        self.num_contradict_paris = 0
        self.num_GlobalContradict_paris = 0

    def add_phrase(self, phrase):
        if phrase not in self.phrase2idx:
            self.phrase2idx[phrase] = self.idx
            self.idx2phrase[self.idx] = phrase
            self.idx += 1

    def add_contradict(self, contract_pair):
        contract_pair = contract_pair.strip().split('<->')
        ori_phrase_str = contract_pair[0]
        ori_phrase = ori_phrase_str.replace('_',' ')
        ori_phrase_idx = None
        if ori_phrase in self.phrase2idx:
            ori_phrase_idx = self.phrase2idx[ori_phrase]
        contract_phrase_str = contract_pair[1]
        contract_phrases = contract_phrase_str.split(',')
        contract_idxes = []
        icontra_ori_list= []
        for icontra in contract_phrases:
            icontra_ori = icontra.replace('_',' ')
            icontra_ori_list.append(icontra_ori)
            if icontra_ori in self.phrase2idx:
                idx = self.phrase2idx[icontra_ori]
                contract_idxes.append(idx)
        if (not ori_phrase_idx is None)&(len(contract_idxes)>0):
            self.idx2contractIdx[ori_phrase_idx] = contract_idxes
            self.phrase2contractphrase[ori_phrase] = ','.join(icontra_ori_list)
            self.num_contradict_paris += 1
        else:
            print('error in adding contradiction:%s'%ori_phrase)


    def add_global_contradict(self, contract_pair):
        contract_pair = contract_pair.strip().split('<->')
        ori_phrase_str = contract_pair[0]
        ori_phrase = ori_phrase_str.replace('_',' ')
        ori_phrase_idx = None
        if ori_phrase in self.phrase2idx:
            ori_phrase_idx = self.phrase2idx[ori_phrase]
        contract_phrase_str = contract_pair[1]
        contract_phrases = contract_phrase_str.split(',')
        contract_idxes = []
        icontra_ori_list= []
        for icontra in contract_phrases:
            icontra_ori = icontra.replace('_',' ')
            icontra_ori_list.append(icontra_ori)
            if icontra_ori in self.phrase2idx:
                idx = self.phrase2idx[icontra_ori]
                contract_idxes.append(idx)
        if (not ori_phrase_idx is None)&(len(contract_idxes)>0):
            self.idx2GlobalContractIdx[ori_phrase_idx] = contract_idxes
            self.phrase2GlobalContractphrase[ori_phrase] = ','.join(icontra_ori_list)
            self.num_GlobalContradict_paris += 1
        else:
            print('error in adding contradiction:%s'%ori_phrase)

    def __call__(self, phrase):
        if phrase not in self.phrase2idx:
            return self.phrase2idx['<unk>']
        return self.phrase2idx[phrase]

    def __len__(self):
        return len(self.phrase2idx)



