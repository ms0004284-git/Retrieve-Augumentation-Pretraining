from datasets import load_dataset
import json
dataset = load_dataset("sciq")

print(len(dataset['train']))
training_data = []
validation_data = []
testing_data = []

for train in dataset['train']:
    training_data.append(train)
for valid in dataset['validation']:
    validation_data.append(valid)
for test in dataset['test']:
    testing_data.append(test)


with open('/user_data/Cloze/dataset/sciq/sciq_train.json', "w", encoding="utf-8") as f:
    json.dump(training_data, f)
with open('/user_data/Cloze/dataset/sciq/sciq_valid.json', "w", encoding="utf-8") as f:
    json.dump(validation_data, f)
with open('/user_data/Cloze/dataset/sciq/sciq_test.json', "w", encoding="utf-8") as f:
    json.dump(testing_data, f)