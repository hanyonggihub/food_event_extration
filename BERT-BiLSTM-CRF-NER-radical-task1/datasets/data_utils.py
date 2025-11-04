import jsonlines
import re
import pandas as pd
import random
def get_dicts(BIO_file):
    BIO_dict={}
    data2 = pd.DataFrame(pd.read_csv(BIO_file, encoding='gbk',dtype='object'))
    for index, row in data2.iterrows():
        this_tag=row['tag']
        this_B_tag=row['B-tag']
        this_I_tag=row['I-tag']
        BIO_dict[this_tag]=[this_B_tag,this_I_tag]
    return BIO_dict

def split_train_test_valid(all_data,split_size):
    rows_len=len(all_data)
    print(rows_len)
    train_text_token = []
    train_label = []
    train_trigger_pos = []
    # dev token
    dev_text_token = []
    # dev bio
    dev_label = []
    dev_trigger_pos = []
    # test token
    test_text_token = []
    # test bio
    test_label = []
    test_trigger_pos = []
    for index in range(rows_len):
        if index < int(rows_len * split_size[0]):
            print("train" + str(index))
            # token
            train_text_token.append(all_data[index][0])
            # label
            train_label.append(all_data[index][1])
        # dev
        elif index >= int(rows_len * split_size[0]) and index < (
                int(rows_len * split_size[0]) + int(rows_len * split_size[1])):
            print("dev" + str(index))
            # token
            dev_text_token.append(all_data[index][0])
            # label
            dev_label.append(all_data[index][1])

        else:
            print("test" + str(index))
            # token
            test_text_token.append(all_data[index][0])
            # label
            test_label.append(all_data[index][1])

    with open("..\\datasets\\train\\text_token", "a", encoding='utf-8') as f:  # ,encoding='utf-8'
        for index, this in enumerate(train_text_token):
            f.write(this + '\n')
    with open("..\\datasets\\train\\label", "a", encoding='utf-8') as f:  # ,encoding='utf-8'
        for index, this in enumerate(train_label):
            f.write(this + '\n')
    with open("..\\datasets\\dev\\text_token", "a", encoding='utf-8') as f:  # ,encoding='utf-8'
        for this in dev_text_token:
            f.write(this + '\n')
    with open("..\\datasets\\dev\\label", "a", encoding='utf-8') as f:  # ,encoding='utf-8'
        for this in dev_label:
            f.write(this + '\n')
    with open("..\\datasets\\test\\text_token", "a", encoding='utf-8') as f:  # ,encoding='utf-8'
        for this in test_text_token:
            f.write(this + '\n')
    with open("..\\datasets\\test\\label", "a", encoding='utf-8') as f:  # ,encoding='utf-8'
        for this in test_label:
            f.write(this + '\n')

def data_process(filename,BIO_dict):
    text_list=[]
    tag_list=[]
    dict_list = []
    with open(filename, 'r', encoding='utf-8') as f:
        for item in jsonlines.Reader(f):
            dict_list.append(item)
    for this_dict in dict_list:
        this_text=this_dict['text']
        this_events=this_dict['event_list']
        for this_event in this_events:
            this_event_name=this_event['event_type']
            this_bio=BIO_dict[this_event_name]
            this_BIO=['O']*len(this_text)
            this_trigger=this_event['trigger']
            for this_pos in re.finditer(this_trigger,this_text):
                poses=this_pos.span()
                this_BIO[poses[0]]=this_bio[0]
                for index in range(poses[0]+1,poses[1]):
                    this_BIO[index]=this_bio[1]
            text_list.append(this_text)
            tag_list.append((' ').join(this_BIO))
    all_data=list(map(list,zip(text_list,tag_list)))
    return all_data


if __name__ == '__main__':
    BIO_file = 'BIO_tags.csv'
    split_size = [0.7, 0.1, 0.2]  # train dev test
    BIO_dict = get_dicts(BIO_file)
    all_data = data_process('all_data.jsonl', BIO_dict)
    split_train_test_valid(all_data, split_size)