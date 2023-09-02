import re
import json
from random import choice
from pyserini.search.lucene import LuceneSearcher

def extract_sentences_with_word(text, word):
    pattern = r'\b[^.?!]*\b{}\b[^.?!]*[.?!]'.format(word)
    sentences = re.findall(pattern, text)

    return sentences

def get_ans_dtt_pairs():
    with open(f'/user_data/Cloze/dataset/mcq/total_new_cleaned_train.json', "r", encoding="utf-8") as f:
        data = json.load(f)
        return data

# search_word = 'enjoy'
not_found_word = set()
noise_word = []
ans_dtt_pairs = get_ans_dtt_pairs()
training_data = []
for pair in ans_dtt_pairs:
    search_word = pair['answer']
    searcher = LuceneSearcher('/user_data/wikidata/sample_collection_jsonl')
    hits = searcher.search(search_word)
    sentence = []
    for i in range(len(hits)):
        
        try:
            sentence += extract_sentences_with_word(json.loads(hits[i].raw)['contents'], search_word)
        except:
            noise_word.append(search_word)

            
        if len(sentence) == 9:
            break
    
    # retrive 不到的 word
    
    if len(sentence) < 9:
        not_found_word.add(search_word)
        # print(search_word)

    while len(sentence) > 0:
        s = choice(sentence)
        if len(sentence) > 6 :
            training_data.append(
                {
                    'sentence' : re.sub(fr'\b{search_word}\b', '<extra_id_0>', s, count=1),
                    'label' : '<pad> <extra_id_0> {} <extra_id_1>'.format(pair['distractors'][0]) #only a distractor
                }
            )
        elif len(sentence) > 3:
            training_data.append(
                {
                    'sentence' : re.sub(fr'\b{search_word}\b', '<extra_id_0>', s, count=1),
                    'label' : '<pad> <extra_id_0> {} <extra_id_1>'.format(pair['distractors'][1])
                }
            )
        else:
            try:
                training_data.append(
                    {
                        'sentence' : re.sub(fr'\b{search_word}\b', '<extra_id_0>', s, count=1),
                        'label' : '<pad> <extra_id_0> {} <extra_id_1>'.format(pair['distractors'][2])
                    }
                )
            except:
                print(f'Label number error: {pair}')
        sentence.remove(s)
    print(f'Trainging data num:{len(training_data)}')
    


print(f'no result search word:{not_found_word}')
print(f'noise search word:{noise_word}')
with open('/user_data/Cloze/dataset/mcq/mask_sentence_with_dtt/mcq_mask_sentence_with_dtt_for_t5.json', "w", encoding="utf-8") as f:
    json.dump(training_data, f)
        
        

