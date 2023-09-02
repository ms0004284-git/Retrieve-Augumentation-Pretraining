import re
import json
from random import choice
from pyserini.search.lucene import LuceneSearcher

def extract_sentences_with_word(text, word):
    pattern = r'\b[^.?!]*\b{}\b[^.?!]*[.?!]'.format(word)
    sentences = re.findall(pattern, text)

    return sentences

def get_ans_dtt_pairs():
    with open(f'/user_data/Cloze/dataset/sciq/sciq_total_filter_mcq_overlap_de.json', "r", encoding="utf-8") as f:
        data = json.load(f)
        return data

not_found_word = set()
noise_word = []
ans_dtt_pairs = get_ans_dtt_pairs()
training_data = []
for pair in ans_dtt_pairs:
    search_word = pair['correct_answer']
    searcher = LuceneSearcher('/user_data/wikidata/sample_collection_jsonl')
    hits = searcher.search(search_word)
    sentence = []
    for i in range(len(hits)):
        
        try:
            answer_sentence = extract_sentences_with_word(json.loads(hits[i].raw)['contents'], search_word)
            
        except:
            noise_word.append(search_word)
        for a_s in answer_sentence:
            sentence.append(a_s)
            
            if len(sentence) >= 9:
                break
        if len(sentence) >= 9:
            break
    
    # retrive 不到的 word
    
    if len(sentence) < 9:
        not_found_word.add(search_word)
        # print(search_word)

    for s in sentence:
        try:
            training_data.append(
                    {
                        'sentence' : re.sub(fr'\b{search_word}\b', '<extra_id_0>', s, count=1),
                        'label' : '<pad> <extra_id_0> {first_dtt} <extra_id_1> {second_dtt} <extra_id_2> {thrid_dtt} <extra_id_3>'
                        .format(
                                    first_dtt=pair['distractor1'],
                                    second_dtt=pair['distractor2'],
                                    thrid_dtt=pair['distractor3']
                                )
                    }
                )
        except:
            pass

    print(f'Trainging data num:{len(training_data)}')
    

print(f'no result search word:{not_found_word}')
print(f'noise search word:{noise_word}')
with open('/user_data/Cloze/dataset/sciq/mask_sentence_with_dtt/t5_new/sciq_all_mask_sentence_with_dtt_for_t5(3dtt).json', "w", encoding="utf-8") as f:
    json.dump(training_data, f)
        
        

