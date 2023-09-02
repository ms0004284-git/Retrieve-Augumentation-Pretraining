# Retrieve-Augumentation-Pretraining
整個訓練包含兩個階段 RAP 的預訓練跟下游的 fine-tuning
## RAP pretraining (Cloze/)
```
python Cloze/trainer.py
```
可以直接開始訓練，預設會連到 WAB 可以辦自己的帳號
Cloze/core/config.py 可以更改超參數以及訓練資料的位置

訓練資料可以透過 Cloze/data_process 下的 py 檔生成對應的資料

## Fine-tuning (CTG/)
這裡主要是使用 RAP pretraining 訓練好的 .ckpt 來 fine-tuning，是直接使用之前學姊沿用的腳本

因為每個方法我都會訓練4個 epoch，我每一個都會拿去 fine-tuning，挑分數最好的當作該方法的實驗結果，所以每個方法下都會有4個.ipynb

CTG/train/DG/parameter_analysis 下有對兩種參數調整進行分析，一個是調整檢索句子的數量 k，另一個是調整 Sciq 訓練資料量對實驗結果的影響


## wikidata
wikidata 太大沒辦法放上來，所以要重新建立 wikidata 的反向索引
可以參考 https://github.com/castorini/pyserini/blob/master/docs/usage-index.md#building-a-bm25-index-embeddable-python-implementation 
其中 Building a BM25 Index 下的製作方式
我是使用 https://dumps.wikimedia.org/enwiki/latest/ 中
enwiki-latest-pages-articles-multistream.xml.bz2 的檔案來建立反向索引
