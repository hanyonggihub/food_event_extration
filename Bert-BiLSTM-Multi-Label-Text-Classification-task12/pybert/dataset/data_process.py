import jsonlines
import re
import pandas as pd
import random
event_type=['假冒','超标']

def data_process(filename):
    all_infor=[]
    dict_list = []
    with open(filename, 'r', encoding='utf-8') as f:
        for item in jsonlines.Reader(f):
            dict_list.append(item)
    for this_dict in dict_list:
        this_text=this_dict['text']
        this_events=this_dict['event_list']
        for this_event in this_events:
            this_infor=[]
            this_event_name=this_event['event_type']
            this_trigger=this_event['trigger']
            this_infor.append(this_text)
            for this_pos in re.finditer(this_trigger,this_text):
                poses=this_pos.span()
                this_infor.append(('_').join(list(map(str,list(poses)))))
                break
            for index,this_event_type in enumerate(event_type):
                if this_event_name==this_event_type:
                    this_infor.append(index)
            all_infor.append(this_infor)
    this_df = pd.DataFrame(all_infor, columns=['摘要','触发词','类别'])
    this_df.to_excel('all_data.xls')


if __name__ == '__main__':
    all_data = data_process('all_data.jsonl')