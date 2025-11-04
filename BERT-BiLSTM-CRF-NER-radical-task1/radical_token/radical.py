# encoding=utf-8

import re
import csv
import urllib.request
from bs4 import BeautifulSoup
import pandas as pd

class Radical(object):
    dictionary_filepath = r'D:\nlp\事件抽取2\BERT-BiLSTM-CRF-NER-radical-task1\radical_token\xinhua.csv'
    baiduhanyu_url = 'http://hanyu.baidu.com/zici/s?ptype=zici&wd=%s'

    def __init__(self):
        self.read_dictionary()

        self.origin_len = len(self.dictionary)

    def read_dictionary(self):
        self.dictionary = {}

        file = open(self.dictionary_filepath,'r',encoding="utf-8")
        reader = csv.reader(file)

        for line in reader:
            self.dictionary[line[0]]= line[1]

        file.close()

    def write_dictionary(self):
        with open(self.dictionary_filepath,'w', encoding='utf-8') as f:
            csv_write = csv.writer(f)
            for word in self.dictionary:
                csv_write.writerow([word, self.dictionary[word]])

    def get_radical(self,word):
        #word = word.decode('utf-8')
        if word in self.dictionary:
            return self.dictionary[word]
        else:
            return self.get_radical_from_baiduhanyu(word)

    def post_baidu(self,url):
        #print(url)
        try:
            timeout = 5
            request = urllib.request.Request(url)
            #伪装HTTP请求
            request.add_header('User-agent', 'Mozilla/5.0')
            request.add_header('connection','keep-alive')
            request.add_header('referer', url)
            # request.add_header('Accept-Encoding', 'gzip')  # gzip可提高传输速率，但占用计算资源
            response = urllib.request.urlopen(request, timeout = timeout)
            html = response.read()
            #if(response.headers.get('content-encoding', None) == 'gzip'):
            #    html = gzip.GzipFile(fileobj=StringIO.StringIO(html)).read()
            response.close()
            return html
        except Exception as e:
            print('URL Request Error:'+str(e))
            return None

    def anlysis_radical_from_html(self,html_doc):
        soup = BeautifulSoup(html_doc, 'html.parser')
        try:
            li = soup.find(id="radical")
            radical = li.span.contents[0]
        except:
            radical='0'
        return radical

    def add_in_dictionary(self,word,radical):
        # add in file
        file_object = open(self.dictionary_filepath,'a+')
        file_object.write(word+','+radical+'\r\n')
        file_object.close()

        # refresh dictionary
        self.read_in_dictionary()

    def get_radical_from_baiduhanyu(self,word):
        url = self.baiduhanyu_url % urllib.parse.quote(word)
        html = self.post_baidu(url)

        if html == None:
            return None

        radical = self.anlysis_radical_from_html(html)
        if radical != None:
            self.dictionary[word] = radical

        return radical

    def save(self):
        if len(self.dictionary) > self.origin_len:
            self.write_dictionary()

    #检验是否全是中文字符
    def is_chinese(self,char):
        if not '\u4e00' <= char <= '\u9fa5':
            return False
        return True

    def get_radical_token(self,token_list):
        seq_radical=[]
        #弄一个序列标点符号和字母，标0（占位符）？？，在输入中怎么处理？
        #弄成同一个空格，用同一个表示？
        num_set = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
        num_dict = {'0': '零', '1': '一', '2': '二', '3': '三', '4': '四', '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'}
        for this_token in token_list:
            #如果是数字就转化成文字
            if this_token in num_set:
                seq_radical.append(self.get_radical(num_dict[this_token]))
                continue
            #如果不是字母或标点符号就直接append'0'
            if self.is_chinese(this_token):
                this_radical=self.get_radical(this_token)
                if not this_radical:
                    seq_radical.append('0')
                else:
                    seq_radical.append(this_radical)
                continue
            seq_radical.append('0')
        return seq_radical




    