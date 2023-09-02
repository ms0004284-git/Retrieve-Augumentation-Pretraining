import json

with open(f'/user_data/Cloze/dataset/sciq/sciq_train.json', "r", encoding="utf-8") as f:
    data_train = json.load(f)
with open(f'/user_data/Cloze/dataset/sciq/sciq_valid.json', "r", encoding="utf-8") as f:
    data_valid = json.load(f)
with open(f'/user_data/Cloze/dataset/sciq/sciq_test.json', "r", encoding="utf-8") as f:
    data_test = json.load(f)

sciq_data = data_train + data_valid + data_test

with open(f'/user_data/CTG/data/MCQ/total_new_cleaned_train.json', "r", encoding="utf-8") as f:
    mcq_train_data = json.load(f)
with open(f'/user_data/CTG/data/MCQ/total_new_cleaned_test.json', "r", encoding="utf-8") as f:
    mcq_test_data = json.load(f)
mcq_data = mcq_train_data + mcq_test_data

mcq_all_delete_sciq_overlap = mcq_data
count = 0
not_count = 0
print(f'Before Num:{len(mcq_all_delete_sciq_overlap)}')
for sciq in sciq_data:
    for mcq in mcq_data:
        mcq_distractors = set(mcq['distractors'])
        sciq_distractors = set([sciq['distractor1'], sciq['distractor2'], sciq['distractor3']])

        if mcq['answer'] == sciq['correct_answer'] and mcq_distractors == sciq_distractors:
            mcq_all_delete_sciq_overlap.remove(mcq)
            count+=1
            

print(f'Remove Num:{count}')
print(f'After Num:{len(mcq_all_delete_sciq_overlap)}')


with open('/user_data/Cloze/dataset/mcq/mcq_all_filter_sciq_overlap.json', "w", encoding="utf-8") as f:
    json.dump(mcq_all_delete_sciq_overlap, f)