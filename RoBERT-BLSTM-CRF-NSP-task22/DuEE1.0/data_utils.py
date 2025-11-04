import jsonlines
import re
import pandas as pd
import random
import hanlp
import numpy as np

HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH) # 世界最大中文语料库

def get_dicts(BIO_file):
    BIO_dict={}
    data2 = pd.DataFrame(pd.read_csv(BIO_file, encoding='gbk',dtype='object'))
    for index, row in data2.iterrows():
        this_tag=row['tag']
        this_B_tag=row['B-tag']
        this_I_tag=row['I-tag']
        BIO_dict[this_tag]=[this_B_tag,this_I_tag]
    return BIO_dict

def data_process(filename,filename2,bio_dict):
            dict_list=[]
            with open(filename,'r', encoding='utf-8') as f:
                for item in jsonlines.Reader(f):
                    dict_list.append(item)
            with open(filename2,'r', encoding='utf-8') as f:
                for item in jsonlines.Reader(f):
                    dict_list.append(item)
            ##get infors
            trigger_event_list=[]#later, I will make it can use directly.
            BIO_tag_list=[]
            token_list=[]
            NER_tag_list=[]
            for dict in dict_list:
                text=dict['text']
                text=text.strip(' ')
                # if '甲骨文刚刚为旗下 VirtualBox 虚拟机软件推出了' in text:
                #     print(len(text))
                #     print('99999999')

                for this_event in dict['event_list']:
                            event_type=this_event['event_type']
                            #过滤出来event_type有产品行为的
                            if "产品行为" not in event_type:
                                continue
                            trigger=this_event['trigger']
                            arguments=this_event['arguments']
                            this_token=text
                            #start at 0 end at +1
                            # event_type_len=len(event_type)
                            # this_ner_list.extend([[i.start()+event_type_len,i.end()+event_type_len] for i in re.finditer(trigger,text)])
                            this_BIO_tag=["O"] * len(text)
                            for argument in arguments:
                                # start at 0 end at +1
                                bio_tags=bio_dict[argument['role']]
                                argument_start_index=int(argument['argument_start_index'])
                                argument_end_index=argument_start_index+len(argument['argument'])
                                this_BIO_tag[argument_start_index]=bio_tags[0]
                                for index in range(argument_start_index+1,argument_end_index):
                                    this_BIO_tag[index]= bio_tags[1]
                            # fill_str=" ".join(['O']*len(event_type))

                            this_BIO_tag2=" ".join(this_BIO_tag)
                            #hanlp NER
                            this_NER_tag=['O']*len(this_token)
                            output_dict = HanLP(this_token, tasks=['ner/msra'])
                            output_dict_ner = output_dict['ner/msra']
                            print(output_dict_ner)
                            for i, ner in enumerate(output_dict_ner):
                                this_name = ner[1]
                                if this_name == 'ORGANIZATION' or this_name=='DATE':#or this_name =='LOCATION'
                                    # if '局' in ner[0]:
                                    #     continue
                                    for index in range(ner[2],ner[3]):
                                        this_NER_tag[index]=this_name
                            this_NER_tag2=" ".join(this_NER_tag)
                            trigger_event_list.append(event_type+' '+trigger)
                            BIO_tag_list.append(this_BIO_tag2)
                            NER_tag_list.append(this_NER_tag2)
                            token_list.append(this_token)
            all_data=list(map(list,zip(trigger_event_list,token_list,BIO_tag_list,NER_tag_list)))
            return all_data

def split_train_test_valid(all_data,split_size):
    rows_len=len(all_data)
    print(rows_len)
    train_text_token = []
    train_label = []
    train_ner = []
    train_trigger_event=[]

    dev_text_token = []
    dev_label = []
    dev_ner = []
    dev_trigger_event = []

    test_text_token = []
    test_label = []
    test_ner = []
    test_trigger_event = []

    for index in range(rows_len):
        if index < int(rows_len * split_size[0]):
            train_text_token.append(all_data[index][1])
            train_label.append(all_data[index][2])
            train_ner.append(all_data[index][3])
            train_trigger_event.append(all_data[index][0])

        # dev
        elif index >= int(rows_len * split_size[0]) and index < (
                int(rows_len * split_size[0]) + int(rows_len * split_size[1])):
            print("dev" + str(index))
            # token
            dev_text_token.append(all_data[index][1])
            # label
            dev_label.append(all_data[index][2])
            dev_ner.append(all_data[index][3])
            dev_trigger_event.append(all_data[index][0])

        else:
            print("test" + str(index))
            # token
            test_text_token.append(all_data[index][1])
            # label
            test_label.append(all_data[index][2])
            test_ner.append(all_data[index][3])
            test_trigger_event.append(all_data[index][0])
    with open("..\\datasets2\\train\\text_token", "a", encoding='utf-8') as f:  # ,encoding='utf-8'
        for index, this in enumerate(train_text_token):
            split_list=this.split('\n')
            if len(split_list)>1:
                this=(' ').join(split_list)
            f.write(this + '\n')
    with open("..\\datasets2\\train\\label", "a", encoding='utf-8') as f:  # ,encoding='utf-8'
        for index, this in enumerate(train_label):
            f.write(this + '\n')
    with open("..\\datasets2\\train\\ner", "a", encoding='utf-8') as f:  # ,encoding='utf-8'
        for index, this in enumerate(train_ner):
            f.write(this + '\n')
    with open("..\\datasets2\\train\\trigger_event", "a", encoding='utf-8') as f:  # ,encoding='utf-8'
        for index, this in enumerate(train_trigger_event):
            f.write(this + '\n')
    with open("..\\datasets2\\dev\\text_token", "a", encoding='utf-8') as f:  # ,encoding='utf-8'
        for this in dev_text_token:
            split_list = this.split('\n')
            if len(split_list) > 1:
                this = (' ').join(split_list)
            f.write(this + '\n')
    with open("..\\datasets2\\dev\\label", "a", encoding='utf-8') as f:  # ,encoding='utf-8'
        for this in dev_label:
            f.write(this + '\n')
    with open("..\\datasets2\\dev\\ner", "a", encoding='utf-8') as f:  # ,encoding='utf-8'
        for index, this in enumerate(dev_ner):
            f.write(this + '\n')
    with open("..\\datasets2\\dev\\trigger_event", "a", encoding='utf-8') as f:  # ,encoding='utf-8'
        for index, this in enumerate(dev_trigger_event):
            f.write(this + '\n')
    with open("..\\datasets2\\test\\text_token", "a", encoding='utf-8') as f:  # ,encoding='utf-8'
        for this in test_text_token:
            f.write(this + '\n')
    with open("..\\datasets2\\test\\label", "a", encoding='utf-8') as f:  # ,encoding='utf-8'
        for this in test_label:
            f.write(this + '\n')
    with open("..\\datasets2\\test\\ner", "a", encoding='utf-8') as f:  # ,encoding='utf-8'
        for index, this in enumerate(test_ner):
            f.write(this + '\n')
    with open("..\\datasets2\\test\\trigger_event", "a", encoding='utf-8') as f:  # ,encoding='utf-8'
        for index, this in enumerate(test_trigger_event):
            f.write(this + '\n')


if __name__ == '__main__':
        BIO_file= 'BIO_tags.csv'
        BIO_dict=get_dicts(BIO_file)
        all_data=data_process('duee_train.json', 'duee_dev.json', BIO_dict)
        split_size = [0.9, 0.1, 0.0]#tain dev test
        split_train_test_valid(all_data, split_size)
