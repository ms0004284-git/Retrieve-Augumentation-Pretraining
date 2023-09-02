"""
將所有 CLOTH data 分別存儲成 cloth_train、cloth_valid 和 cloth_test 三個檔案
"""

import json
import os, sys
import fnmatch

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

def write_json(data, path):
    
    jsonString = json.dumps(data)
    jsonFile = open(path, "w")
    jsonFile.write(jsonString)
    jsonFile.close()

if __name__ == '__main__':

    train = read_cloth('train')
    valid = read_cloth('valid')
    test = read_cloth('test')

    write_json(train, './data/cloth_train.json')
    write_json(valid, './data/cloth_valid.json')
    write_json(test, './data/cloth_test.json')

    print(len(train), 'train data is writed.')
    print(len(valid), 'valid data is writed.')
    print(len(test), 'test data is writed.')
