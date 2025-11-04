import torch
from pybert.configs.basic_config import config
from pybert.io.bert_processor import BertProcessor
from pybert.model.bert_for_multi_label import BertForMultiLable

def main(text,arch,max_seq_length,do_lower_case):
    processor = BertProcessor(vocab_path=config['bert_vocab_path'], do_lower_case=do_lower_case)
    label_list = processor.get_labels()
    id2label = {i: label for i, label in enumerate(label_list)}
    model = BertForMultiLable.from_pretrained(config['checkpoint_dir'] /f'{arch}', num_labels=len(label_list))
    tokens = processor.tokenizer.tokenize(text)
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[:max_seq_length - 2]
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    input_ids = processor.tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor(input_ids).unsqueeze(0)  # Batch size 1, 2 choices
    logits = model(input_ids)
    probs = logits.sigmoid()
    result = ((probs > 0.50)).float().cpu().numpy()
    labels=processor.get_labels()
    this_text_labels=[]
    index=0
    #转化成关系
    for this in result[0]:
        if this==1.0:
            this_text_labels.append(labels[index])
        index=index+1
    return result[0],this_text_labels

if __name__ == "__main__":
    with open("seqs", "r", encoding='utf8') as f:  # ,encoding='utf-8'
        lines = f.readlines()
        lines = [x.strip() for x in lines]
    seq_label_list=[]
    label_list=[]
    for text in lines:
        max_seq_length = 256
        do_loer_case = True
        arch = 'bert'
        result,this_text_labels= main(text,arch,max_seq_length,do_loer_case)
        for label in this_text_labels:
            label_list.append(label)
            print(label)
            seq_label=label+text[:len(text)-1]+label+text[-1]
            print(seq_label)
            seq_label_list.append(seq_label)
    # with open("labels1", "a", encoding='utf8') as f:  # ,encoding='utf-8'
    #     for this_label in label_list:
    #         f.write(this_label+'\n')
    # with open("seqs_label1", "a", encoding='utf8') as f:  # ,encoding='utf-8'
    #     for this_seq_label in seq_label_list:
    #         f.write(this_seq_label + '\n')
        # lines = [x.strip() for x in lines]

    
      # '''
      # #output
      # [0.98304486 0.40958735 0.9851305  0.04566246 0.8630512  0.07316463]
      # '''
