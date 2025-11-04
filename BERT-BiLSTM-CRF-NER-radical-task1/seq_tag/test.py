import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from .data_utils import NerDataset, get_idx2tag
from .model import BertBilstmCrf
from . import metric


here = os.path.dirname(os.path.abspath(__file__))


def test(hparams):
    device = hparams.device
    seed = hparams.seed
    torch.manual_seed(seed)

    pretrained_model_path = hparams.pretrained_model_path
    radical_pretrained_model_path = hparams.radical_model_dir
    tagset_file = hparams.tagset_file
    model_file = hparams.model_file
    validation_file = hparams.validation_file

    max_len = hparams.max_len
    validation_batch_size = hparams.validation_batch_size

    idx2tag = get_idx2tag(tagset_file)
    hparams.tagset_size = len(idx2tag)
    model = BertBilstmCrf(hparams).to(device)
    model.load_state_dict(torch.load(model_file))
    epoch=0


    if validation_file:
        validation_dataset = NerDataset(validation_file, tagset_path=tagset_file,
                                                    pretrained_model_path=pretrained_model_path,
                                                    radical_pretrain_model_path=radical_pretrained_model_path,
                                                    max_len=max_len, is_train=False)
        val_loader = DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=False)
        model.eval()
        with torch.no_grad():
            tags_true_list = []
            tags_pred_list = []
            for val_i_batch, val_sample_batched in enumerate(tqdm(val_loader, desc='Validation')):
                token_ids = val_sample_batched['token_ids'].to(device)
                token_type_ids = val_sample_batched['token_type_ids'].to(device)
                attention_mask = val_sample_batched['attention_mask'].to(device)
                radical_token_ids = val_sample_batched['radical_token_ids'].to(device)
                radical_token_type_ids = val_sample_batched['radical_token_type_ids'].to(device)
                radical_attention_mask = val_sample_batched['radical_attention_mask'].to(device)
                tag_ids = val_sample_batched['tag_ids'].tolist()
                pred_tag_ids = model.decode(input_ids=token_ids, token_type_ids=token_type_ids,
                                            attention_mask=attention_mask,radical_input_ids=radical_token_ids, radical_token_type_ids=radical_token_type_ids, radical_attention_mask=radical_attention_mask)

                seq_ends = attention_mask.sum(dim=1)
                true_tag_ids = [_tag_ids[:seq_ends[i]] for i, _tag_ids in enumerate(tag_ids)]
                batched_tags_true = [[idx2tag[tag_id] for tag_id in _tag_ids] for _tag_ids in true_tag_ids]
                batched_tags_pred = [[idx2tag[tag_id] for tag_id in _tag_ids] for _tag_ids in pred_tag_ids]
                tags_true_list.extend(batched_tags_true)
                tags_pred_list.extend(batched_tags_pred)

            print(metric.classification_report(tags_true_list, tags_pred_list))
            f1 = metric.f1_score(tags_true_list, tags_pred_list)
            precision = metric.precision_score(tags_true_list, tags_pred_list)
            recall = metric.recall_score(tags_true_list, tags_pred_list)
            accuracy = metric.accuracy_score(tags_true_list, tags_pred_list)
            print(str(epoch) + "f1:" + str(f1))
            print(str(epoch) + "precision:" + str(precision))
            print(str(epoch) + "recall:" + str(recall))
            print(str(epoch) + "accuracy:" + str(accuracy))