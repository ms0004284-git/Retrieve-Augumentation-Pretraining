import json


with open(f'/user_data/Cloze/dataset/sciq/sciq_train.json', "r", encoding="utf-8") as f:
    sciq_train = json.load(f)
with open(f'/user_data/Cloze/dataset/sciq/sciq_valid.json', "r", encoding="utf-8") as f:
    sciq_valid = json.load(f)
with open(f'/user_data/Cloze/dataset/sciq/sciq_test.json', "r", encoding="utf-8") as f:
    sciq_test = json.load(f)

sciq_data = sciq_train + sciq_valid + sciq_test

with open(f'/user_data/CTG/data/MCQ/total_new_cleaned_train.json', "r", encoding="utf-8") as f:
    mcq_train_data = json.load(f)
with open(f'/user_data/CTG/data/MCQ/total_new_cleaned_test.json', "r", encoding="utf-8") as f:
    mcq_test_data = json.load(f)
mcq_data = mcq_train_data + mcq_test_data


sciq_all_delete_mcq_overlap = sciq_data
count = 0
not_count = 0
print(f'Before Num:{len(sciq_all_delete_mcq_overlap)}')
for sciq in sciq_data:
    for mcq in mcq_data:
        mcq_distractors = set(mcq['distractors'])
        sciq_distractors = set([sciq['distractor1'], sciq['distractor2'], sciq['distractor3']])

        if mcq['answer'] == sciq['correct_answer'] and mcq_distractors == sciq_distractors:
            
            try:
                sciq_all_delete_mcq_overlap.remove(sciq)
                count+=1
            except:
                pass
            
            

print(f'Remove Num:{count}')
print(f'After Num:{len(sciq_all_delete_mcq_overlap)}')


with open('/user_data/Cloze/dataset/sciq/sciq_train_filter_mcq_overlap_de.json', "w", encoding="utf-8") as f:
    json.dump(sciq_all_delete_mcq_overlap, f)





