import json
from sklearn.model_selection import train_test_split
def read_data(item):
    path = '/user_data/CTG/data/MCQ/total_new_cleaned_{}.json'.format(item)
    with open(path) as f:
        data = json.load(f)
    return data

def save_data(item, data):
    path = '/user_data/CTG/data/MCQ/total_new_cleaned_{}_splited.json'.format(item)
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f)
    print(f'{item} set saved')

train = read_data('train')
test = read_data('test')

train, valid = train_test_split(train, random_state=777, train_size=0.9)
print(len(train))
print(len(valid))
print(len(test))

save_data('train', train)
save_data('valid', valid)