import re
import os
import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))

def get_positions(start_index, end_index, token_len,max_len):
    start_index=int(start_index)
    end_index=int(end_index)
    token_len=int(token_len)
    if token_len<=max_len:
        trigger_pos = list(range(-start_index, 0)) + [0] * (end_index - start_index) + list(
            range(1, max_len - end_index + 1))
    else:
        trigger_pos = list(range(-start_index, 0)) + [0] * (end_index - start_index) + list(
            range(1, token_len - end_index + 1))
        trigger_pos=trigger_pos[:max_len]
    return trigger_pos

def read_data(input_file,max_len):##改改
    tokens_list = []
    tags_list = []
    triggers_list=[]
    event_types_list=[]
    ners_list= []
    tagset = set()
    with open(os.path.join(input_file,'text_token'), 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            this_token=[x for x in line]
            tokens_list.append(this_token)
    with open(os.path.join(input_file,'trigger_event'), 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            trigger_events=line.split(' ')
            triggers_list.append(trigger_events[1])
            event_types_list.append(trigger_events[0])
    with open(os.path.join(input_file,'ner'), 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            ners=line.split(' ')
            ners_list.append(ners)
    with open(os.path.join(input_file,'label'), 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            tags=line.split(' ')
            tags_list.append(tags)
            for tag in tags:
                tagset.add(tag)
    tagset = list(tagset)
    tagset.sort()
    return tokens_list,tags_list,ners_list,triggers_list,event_types_list, tagset


def save_tagset(tagset, output_file):
    with open(output_file, 'w', encoding='utf-8') as f_out:
        f_out.write('\n'.join(tagset))


def get_tag2idx(file):
    with open(file, 'r', encoding='utf-8') as f_in:
        tagset = re.split(r'\s+', f_in.read().strip())
    return dict((tag, idx) for idx, tag in enumerate(tagset))


def get_idx2tag(file):
    with open(file, 'r', encoding='utf-8') as f_in:
        tagset = re.split(r'\s+', f_in.read().strip())
    return dict((idx, tag) for idx, tag in enumerate(tagset))


def save_checkpoint(checkpoint_dict, file):
    with open(file, 'w', encoding='utf-8') as f_out:
        json.dump(checkpoint_dict, f_out, ensure_ascii=False, indent=2)


def load_checkpoint(file):
    with open(file, 'r', encoding='utf-8') as f_in:
        checkpoint_dict = json.load(f_in)
    return checkpoint_dict


class NerDataset(Dataset):
    def __init__(self, data_file_path, tagset_path, pretrained_model_path=None, max_len=128, is_train=False,TF_token_b=False):
        self.data_file_path = data_file_path
        self.tagset_path = tagset_path
        self.pretrained_model_path = pretrained_model_path or 'Robert-base-chinese'
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_path)
        self.max_len = max_len
        self.is_train = is_train
        self.TF_token_b=TF_token_b
        self.ner2idx = { 'O': 0, 'ORGANIZATION': 1, 'LOCATION': 2, 'DATE': 3}#'<PAD>': 0, '<UNK>': 1,

        if is_train:
            self.tokens_list, self.tags_list,self.ners_list,self.triggers_list,self.events_types_list, self.tagset = read_data(data_file_path,max_len=self.max_len)
            save_tagset(self.tagset, self.tagset_path)
        else:
            self.tokens_list, self.tags_list,self.ners_list,self.triggers_list,self.events_types_list, _ = read_data(data_file_path,max_len=self.max_len)
        self.tag2idx = get_tag2idx(self.tagset_path)


    def __len__(self):
        return len(self.tags_list)

    def truncate_seq_pair(self,tokens_a,tokens_b,max_length):
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_tokens = self.tokens_list[idx]
        tokens_a = [this_str for this_str in sample_tokens]
        sample_tags = self.tags_list[idx]
        sample_events_types = self.events_types_list[idx]
        sample_triggers = self.triggers_list[idx]
        sample_ners = self.ners_list[idx]
        # encoded = self.tokenizer.encode_plus(sample_tokens, max_length=self.max_len, pad_to_max_length=True)
        # sample_token_ids = encoded['input_ids']
        # sample_token_type_ids = encoded['token_type_ids']
        # sample_attention_mask = encoded['attention_mask']
        # sample_tags = sample_tags[:self.max_len - 2]
        # sample_tags = ['O'] + sample_tags + ['O'] * (self.max_len - len(sample_tags) - 1)
        # sample_tag_ids = [self.tag2idx[tag] for tag in sample_tags]
        event_type_len = len(sample_events_types)

        if self.TF_token_b:
            tokens_b = self.tokenizer.tokenize(sample_triggers)
            self.truncate_seq_pair(tokens_a, tokens_b, max_length=self.max_len - 3 - event_type_len * 2)
            if len(tokens_a) < len(sample_ners):
                sample_ners = sample_ners[:len(tokens_a)]
                sample_tags = sample_tags[:len(tokens_a)]
        else:
            # Account for [CLS] and [SEP] with '-2'
            if len(tokens_a) > self.max_len - 2- event_type_len * 2:
                tokens_a = tokens_a[:self.max_len - 2- event_type_len * 2]
                sample_ners = sample_ners[:self.max_len - 2- event_type_len * 2]
                sample_tags = sample_tags[:self.max_len - 2- event_type_len * 2]
        padding1 = ['O'] * event_type_len
        event_type = [this_str for this_str in sample_events_types]
        tokens_a = event_type + tokens_a + event_type
        sample_ners = padding1 + sample_ners + padding1
        sample_tags = padding1 + sample_tags + padding1

        tokens = ['[CLS]'] + tokens_a + ['[SEP]']
        sample_ners = ['O'] + sample_ners + ['O']
        sample_tags = ['O'] + sample_tags + ['O']
        segment_ids = [0] * len(tokens)
        if self.TF_token_b:
            tokens += tokens_b + ['[SEP]']
            segment_ids += [1] * (len(tokens_b) + 1)
            sample_tags+=['O'] * (len(tokens_b) + 1)
            sample_ners+=['O'] * (len(tokens_b) + 1)


        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        padding = [0] * (self.max_len - len(input_ids))
        padding2=['O'] * (self.max_len - len(input_ids))

        input_ids += padding
        input_mask += padding
        segment_ids += padding
        sample_tags += padding2
        sample_ners += padding2

        tags_ids = [self.tag2idx[tag] for tag in sample_tags]
        ners_ids = [self.ner2idx[ner] for ner in sample_ners]

        assert len(input_ids) == self.max_len
        assert len(input_mask) == self.max_len
        assert len(segment_ids) == self.max_len
        assert len(ners_ids) == self.max_len
        assert len(tags_ids) == self.max_len

        # sample_input_ids = torch.tensor(input_ids, dtype=torch.long)
        # sample_input_mask = torch.tensor(input_mask, dtype=torch.long)
        # sample_segment_ids = torch.tensor(segment_ids, dtype=torch.long)

        # sample_input_lens = torch.tensor(input_len, dtype=torch.long)

        sample = {
            'token_ids': torch.tensor(input_ids),
            'token_type_ids': torch.tensor(segment_ids),
            'attention_mask': torch.tensor(input_mask),
            'tag_ids': torch.tensor(tags_ids),
            'ner_ids': torch.tensor(ners_ids),
        }
        return sample
