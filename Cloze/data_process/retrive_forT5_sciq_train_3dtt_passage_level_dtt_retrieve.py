import re
import json
from pyserini.search.lucene import LuceneSearcher
import nltk

# 載入nltk的Punkt tokenizer
nltk.download('punkt')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def get_passages_with_word(text, word):

    # 使用Punkt tokenizer將文章分成句子
    passages = tokenizer.tokenize(text)

    # 將句子分成3句一組
    n = 3
    passages = [''.join(passages[i:i+n]) for i in range(0, len(passages), n)]
    
    result = []
    for passage in passages:
        # 邏輯要再想一下
        if word in passage:
            result.append(passage)

    return result

def get_ans_dtt_pairs():
    with open(f'/user_data/Cloze/dataset/sciq/mask_sentence_with_dtt/sciq_train_filter_mcq_overlap.json', "r", encoding="utf-8") as f:
        data = json.load(f)
        return data

not_found_word = set()
noise_word = []
ans_dtt_pairs = get_ans_dtt_pairs()
training_data = []
for pair in ans_dtt_pairs:
    search_word = [pair['answer'], pair['distractor'][0], pair['distractor'][1], pair['distractor'][2]]
    for sw in search_word:
        searcher = LuceneSearcher('/user_data/wikidata/sample_collection_jsonl')
        hits = searcher.search(sw)
        passages = []
        for i in range(len(hits)):
            
            try:
                passages += get_passages_with_word(json.loads(hits[i].raw)['contents'], sw)
            except:
                noise_word.append(sw)

                
            if len(passages) == 9: # 總共 12
                break
        
        # retrive 不到的 word
        
        if len(passages) < 3:
            not_found_word.add(sw)
            # print(sw)
        label_set = list(search_word)
        label_set.remove(sw)
        for p in passages:
            try:
                training_data.append(
                        {
                            'sentence' : re.sub(fr'\b{sw}\b', '<extra_id_0>', p, count=1), # count= 取代幾個 沒有放就是全部
                            'label' : '<pad> <extra_id_0> {first_dtt} <extra_id_1> {second_dtt} <extra_id_2> {thrid_dtt} <extra_id_3>'
                            .format(
                                        first_dtt=label_set[0],
                                        second_dtt=label_set[1],
                                        thrid_dtt=label_set[2]
                                    )
                        }
                    )
            except:
                pass


    print(f'Trainging data num:{len(training_data)}')
    

print(f'no result search word:{not_found_word}')
print(f'noise search word:{noise_word}')
with open('/user_data/Cloze/dataset/sciq/mask_sentence_with_dtt/sciq_train_mask_passage_with_dtt_for_t5(3dtt)_dtt_retrieve(9).json', "w", encoding="utf-8") as f:
    json.dump(training_data, f)
        
        
