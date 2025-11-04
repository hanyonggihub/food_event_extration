import os
import torch
from .data_utils import get_idx2tag
from .model import BertBilstmCrf
from .metric import get_entities
from transformers import BertTokenizer

here = os.path.dirname(os.path.abspath(__file__))


def truncate_seq_pair(tokens_a, tokens_b, max_length):
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
##修改！！看有ner和无ner两个的实际效果
def predict(hparams):
    # all_events_roles_dict = {"fake": ['BRA', 'CAT', 'FCO', 'PRO'],
    #                          "Exce": [ 'BRA', 'CAT', 'COM', 'PRO', 'RIS'],
    #                          "recall": ['BRA', 'CAT', 'COM', 'PRO', 'RRS', 'NUM']}
    # event_list = list(all_events_roles_dict.keys())
    bio2name={'TIM':'时间','RED':'发布产品','WIP':'获奖人','EXH':'上映方','OSD':'下架产品','REC':'召回内容','PUP':'发布方','AWA':'奖项','OMV':'上映影视','RDP':'被下架方','REP':'召回方','AWO':'颁奖机构','OSP':'下架方'}
    ner2idx = {'O': 0, 'ORGANIZATION': 1, 'LOCATION': 2, 'DATE': 3}  # '<PAD>': 0, '<UNK>': 1,
    #read risk_material.txt
    # risk_materials=[]
    # with open(hparams.risk_txt_file,'r') as f:
    #     lines=f.readlines()
    #     for line in lines:
    #         line=line.strip()
    #         risk_materials.append(line)
    device = hparams.device
    seed = hparams.seed
    torch.manual_seed(seed)

    pretrained_model_path = hparams.pretrained_model_path
    tagset_file = hparams.tagset_file
    model_file = hparams.model_file

    idx2tag = get_idx2tag(tagset_file)
    hparams.tagset_size = len(idx2tag)
    model = BertBilstmCrf(hparams).to(device)
    model.load_state_dict(torch.load(model_file,map_location=torch.device('cpu')),strict=False)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_path)
    events_types_list= []
    tokens_list=[]
    tags_list=[]
    ners_list=[]
    triggers_list=[]

    delone_char_list=['EXF','FAB','FAF','PLA','STM']
    with open(hparams.test_input_path+'/text_token', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            # event_type_text = line.split('\t')
            # events_types_list.append([int(event_type_text[1]) for x in range(hparams.max_len)])
            tokens_list.append(line)
    with open(hparams.test_input_path+'/label', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            tags_list.append(line.split(' '))
    with open(hparams.test_input_path+'/ner', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            ners_list.append(line.split(' '))
    with open(hparams.test_input_path+'/trigger_event', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            trigger_event=line.split(' ')
            triggers_list.append(trigger_event[1])
            events_types_list.append(trigger_event[0])

    triple_results = []
    triple_true = []
    for index in range(len(tokens_list)):
        # this_text=texts_list[index]
        # # print(this_text)
        # tokens=[x for x in this_text]
        # event_types=event_type_list[index]
        # encoded = tokenizer.encode_plus(tokens, max_length=hparams.max_len, pad_to_max_length=True,event_types=event_types, return_tensors='pt')
        # input_ids = encoded['input_ids'].to(device)
        # token_type_ids = encoded['token_type_ids'].to(device)
        # attention_mask = encoded['attention_mask'].to(device)
        sample_tokens = tokens_list[index]
        tokens_a = [this_str for this_str in sample_tokens]
        sample_tags = tags_list[index]
        sample_events_types = events_types_list[index]
        sample_triggers = triggers_list[index]
        sample_ners = ners_list[index]

        # encoded = hparams.tokenizer.encode_plus(sample_tokens, max_length=hparams.max_len, pad_to_max_length=True)
        # sample_token_ids = encoded['input_ids']
        # sample_token_type_ids = encoded['token_type_ids']
        # sample_attention_mask = encoded['attention_mask']
        # sample_tags = sample_tags[:hparams.max_len - 2]
        # sample_tags = ['O'] + sample_tags + ['O'] * (hparams.max_len - len(sample_tags) - 1)
        # sample_tag_ids = [hparams.tag2idx[tag] for tag in sample_tags]
        event_type_len = len(sample_events_types)
        if hparams.TF_token_b:
            tokens_b = tokenizer.tokenize(sample_triggers)
            truncate_seq_pair(tokens_a, tokens_b, max_length=hparams.max_len - 3 - event_type_len * 2)
            if len(tokens_a) < len(sample_ners):
                sample_ners = sample_ners[:len(tokens_a)]
                sample_tags = sample_tags[:len(tokens_a)]
        else:
            # Account for [CLS] and [SEP] with '-2'
            if len(tokens_a) > hparams.max_len - 2 - event_type_len * 2:
                tokens_a = tokens_a[:hparams.max_len - 2 - event_type_len * 2]
                sample_ners = sample_ners[:hparams.max_len - 2 - event_type_len * 2]
                sample_tags = sample_tags[:hparams.max_len - 2 - event_type_len * 2]
        padding1 = ['O'] * event_type_len
        event_type = [this_str for this_str in sample_events_types]
        tokens_a = event_type + tokens_a + event_type
        sample_ners = padding1 + sample_ners + padding1
        sample_tags = padding1 + sample_tags + padding1

        tokens = ['[CLS]'] + tokens_a + ['[SEP]']
        sample_ners = ['O'] + sample_ners + ['O']
        sample_tags = ['O'] + sample_tags + ['O']
        segment_ids = [0] * len(tokens)
        if hparams.TF_token_b:
            tokens += tokens_b + ['[SEP]']
            segment_ids += [1] * (len(tokens_b) + 1)
            sample_tags += ['O'] * (len(tokens_b) + 1)
            sample_ners += ['O'] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        padding = [0] * (hparams.max_len - len(input_ids))
        padding2 = ['O'] * (hparams.max_len - len(input_ids))

        input_ids += padding
        input_mask += padding
        segment_ids += padding
        sample_tags += padding2
        sample_ners += padding2

        # label_true_list = [hparams.tag2idx[tag] for tag in sample_tags]
        label_true_list = sample_tags
        ners_ids = [ner2idx[ner] for ner in sample_ners]

        input_ids=torch.tensor(input_ids).to(device).unsqueeze(0)
        segment_ids=torch.tensor(segment_ids).to(device).unsqueeze(0)
        input_mask=torch.tensor(input_mask).to(device).unsqueeze(0)
        # tags_ids=torch.tensor(tags_ids).to(device)
        ner_ids=torch.tensor(ners_ids).to(device).unsqueeze(0)


        with torch.no_grad():
            tag_ids = model.decode(input_ids=input_ids, token_type_ids=segment_ids,
                                                attention_mask=input_mask,ner_ids=ner_ids)[0]
        tags = [idx2tag[tag_id] for tag_id in tag_ids]

        chunks = get_entities(tags)
        this_result = ""

        # event_role_list=all_events_roles_dict[event_list[int(event_types[0])-1]]

        for chunk_type, chunk_start, chunk_end in chunks:
            # if chunk_type not in event_role_list:
            #     continue
            # # delete CAT\BRA\FCO\FAC\COM just have one char
            # T_F_save=False
            if chunk_type in delone_char_list:
                if chunk_end-chunk_start==0:
                    continue

            # complete risk material
            # if chunk_type=='RIS' or chunk_type=="RRS":
            #     if chunk_end-chunk_start>=1:
            #         this_risks = [s for s in risk_materials if ''.join(tokens[chunk_start - 1: chunk_end]) in s]
            #         if len(this_risks)==1:
            #             this_result = this_result + chunk_type + ' ' + ''.join(
            #                     tokens[chunk_start - 1: chunk_end]) + '|'
            #             T_F_save=True
            #
            # if not T_F_save:
            this_result = this_result + bio2name[chunk_type] + ' ' + ''.join(
                tokens[chunk_start: chunk_end+1]) + '|'
        triple_results.append(this_result)
        # print(this_result)


        chunks2 = get_entities(label_true_list)
        this_result2 = ""
        for chunk_type, chunk_start, chunk_end in chunks2:
            this_result2 = this_result2 + bio2name[chunk_type] + ' ' + ''.join(tokens[chunk_start: chunk_end + 1]) + '|'
        triple_true.append(this_result2)


    with open("triples_test.txt", 'w', encoding='utf-8') as f:
        for this_triples in triple_results:
            f.write(this_triples + '\n')
    with open("triples_true.txt", 'w', encoding='utf-8') as f:
       for this_triples in triple_true:
           f.write(this_triples + '\n')






