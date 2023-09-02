"""
刪除可能出現題號的資料集
"""

import re
import json
import os, sys
import fnmatch

def count_question_number(article):
    article_list = article.split()
    number_cnt = 0
    for k in range(len(article_list)):
        if article_list[k] == '_':
            if bool(re.search(r'\d', article_list[k-1])):
                number_cnt+=1
    return number_cnt

def filter_q_article(data):

    reserve_data = []
    remove_data_source = []
    for idx in range(len(data)):
        article = data[idx]['article']
        options = data[idx]['options']
        answers = data[idx]['answers']
        source = data[idx]['source']

        cnt_ = article.count('_')
        q_number_cnt = count_question_number(article)
        if q_number_cnt != 0 or cnt_ == 0:
            remove_data_source.append(source)
        else:
            reserve_data.append(data[idx])
    return reserve_data, remove_data_source

def write_json(data, item):
    path = './data/CLOTH-F/filter_cloth_{}.json'.format(item)
    jsonString = json.dumps(data)
    jsonFile = open(path, "w")
    jsonFile.write(jsonString)
    jsonFile.close()

def get_json_file_list(data_dir):
    files = []
    for root, dir_names, file_names in os.walk(data_dir):
        for filename in fnmatch.filter(file_names, '*.json'):
            files.append(os.path.join(root, filename))
    return files

def read_cloth(item):
    data_dir = './data/CLOTH/{}'.format(item)
    file_list = get_json_file_list(data_dir)
    
    dataset = []
    for file_name in file_list:
        data = json.loads(open(file_name, 'r').read())
        article = data['article']
        ops = data['options']
        ans = data['answers']
        source = data['source']
        
        dataset.append({'article': article, 'options': ops, 'answers': ans, 'source': source})

    return dataset

if __name__ == '__main__':

    data_collections = ['train', 'valid', 'test']
    for item in data_collections:
        data = read_cloth(item)
        reserve_data, remove_data_source = filter_q_article(data)
        write_json(reserve_data, item)

        print(item)
        print('data size: {}'.format(len(data)))
        print('reserve data size: {}'.format(len(reserve_data)))
        print('remove data size: {}'.format(len(remove_data_source)))
        # print('remove data source: {}'.format('\n'.join(remove_data_source)))
        print()
