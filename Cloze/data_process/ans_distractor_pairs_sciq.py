import json

with open(f'/user_data/Cloze/dataset/sciq/sciq_train.json', "r", encoding="utf-8") as f:
    training_data = json.load(f)
print(len(training_data))
data = training_data
print(len(data))
ans_dtt_pairs = []
for d in data:
    option_answer = d['correct_answer']
    option_distractor = [d['distractor1'],d['distractor2'],d['distractor3']]
    ans_dtt_pairs.append(
        {
            'answer' : option_answer,
            'distractor' : option_distractor
        }
    )
with open('/user_data/Cloze/dataset/sciq/mask_sentence_with_dtt/sciq_train_ans_dtt_pair.json', "w", encoding="utf-8") as f:
    json.dump(ans_dtt_pairs, f)

print(len(ans_dtt_pairs))