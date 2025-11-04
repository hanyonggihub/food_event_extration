#encoding:utf-8
import torch
import numpy as np
from ..common.tools import model_device
from ..callback.progressbar import ProgressBar
from pybert.io.bert_processor import BertProcessor
from pybert.configs.basic_config import config

class Predictor(object):
    def __init__(self,model,logger,n_gpu,epoch_metrics,batch_metrics,criterion):
        self.model = model
        self.logger = logger
        self.model, self.device = model_device(n_gpu= n_gpu, model=self.model)
        self.epoch_metrics = epoch_metrics
        self.batch_metrics = batch_metrics
        self.start_epoch = 1
        self.global_step = 0
        self.criterion = criterion

    def epoch_reset(self):
        self.outputs = []
        self.targets = []
        self.result = {}
        for metric in self.epoch_metrics:
            metric.reset()

    def test_epoch(self,data):
        pbar = ProgressBar(n_total=len(data),desc="Evaluating")
        self.epoch_reset()
        for step, batch in enumerate(data):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                input_ids, input_mask, segment_ids, label_ids, triggers = batch
                logits = self.model(input_ids, segment_ids, input_mask, triggers=triggers)
            self.outputs.append(logits.cpu().detach())
            self.targets.append(label_ids.cpu().detach())
            pbar(step=step)
        self.outputs = torch.cat(self.outputs, dim = 0).cpu().detach()
        self.targets = torch.cat(self.targets, dim = 0).cpu().detach()
        # print("targets："+str(self.targets))
        # print("outputs："+str(self.outputs))
        loss = self.criterion(target = self.targets, output=self.outputs)
        self.result['test_loss'] = loss.item()
        print("------------- test result --------------")
        if self.epoch_metrics:
            for metric in self.epoch_metrics:
                metric(logits=self.outputs, target=self.targets)
                value = metric.value()
                if value:
                    self.result[f'test_{metric.name()}'] = value
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        print("test_result："+str(self.result))
        return self.result

    def predict(self,data):
        pbar = ProgressBar(n_total=len(data),desc='Testing')
        all_onehot = None
        for step, batch in enumerate(data):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                # input_ids, input_mask, segment_ids, label_ids = batch
                # logits = self.model(input_ids, segment_ids, input_mask)
                input_ids, input_mask, segment_ids, label_ids, triggers = batch
                logits = self.model(input_ids, segment_ids, input_mask,triggers=triggers)
                ##改改改成softmax
                logits = logits.sigmoid()
            if all_onehot is None:
                all_onehot = ((logits > 0.5)).float().cpu().numpy()
            else:
                all_onehot = np.concatenate([all_onehot,((logits > 0.5)).float().cpu().numpy()],axis = 0)
            pbar(step=step)
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        processor = BertProcessor(vocab_path=config['bert_vocab_path'], do_lower_case=True)
        labels=processor.get_labels()
        texts_labels_list=[]
        for this_onehot in all_onehot:
            index=0
            labels_list=[]
            for this in this_onehot:
                if this==1.0:
                    labels_list.append(labels[index])
                index=index+1
            texts_labels_list.append(labels_list)
        test_log = self.test_epoch(data)
        # print(test_log)
        logs = dict(test_log,)
        # print(logs)
        show_info ="-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])
        # self.logger.info(show_info)
        # print(show_info)
        return all_onehot, texts_labels_list



