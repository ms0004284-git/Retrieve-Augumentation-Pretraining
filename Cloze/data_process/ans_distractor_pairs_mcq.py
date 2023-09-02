import json

ans_dtt_pairs = []
for i in range(2341):
    with open(f'/user_data/Cloze/dataset/mcq/total_new_cleaned_train.json', "r", encoding="utf-8") as f:
        data = json.load(f)

    for option, answer in zip(data['options'], data['answers']):
        if answer == 'A':
            option_answer = option[0]
            option_distractor = [x for x in option if x != option_answer]
        elif answer == 'B':
            option_answer = option[1]
            option_distractor = [x for x in option if x != option_answer]
        elif answer == 'C':
            option_answer = option[2]
            option_distractor = [x for x in option if x != option_answer]
        elif answer == 'D':
            option_answer = option[3]
            option_distractor = [x for x in option if x != option_answer]
        else:
            raise

        ans_dtt_pairs.append(
            {
                'answer' : option_answer,
                'distractor' : option_distractor
            }
        )
with open('/user_data/Cloze/dataset/cloth/ans_dtt_pairs/only_middle.json', "w", encoding="utf-8") as f:
    json.dump(ans_dtt_pairs, f)