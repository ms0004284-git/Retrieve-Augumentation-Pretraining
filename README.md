# Retrieve-Augmentation-Pretraining
The entire training process includes two stages: RAP pretraining and downstream fine-tuning.
## RAP Pretraining (Cloze/)
```
python Cloze/trainer.py
```
You can directly start training using this command. By default, it will connect to WAB. You can register your own account.

Hyperparameters and training data paths can be modified in `Cloze/core/config.py`.

Training data can be generated using the Python scripts under `Cloze/data_process`.

## Fine-tuning (CTG/)

This stage mainly uses the `.ckpt` files trained from RAP pretraining for fine-tuning, following the scripts previously used by our senior.

Since I train each method for 4 epochs and fine-tune each checkpoint, I will select the one with the best score as the experimental result for that method. Therefore, under each method, there will be 4 `.ipynb` files.

In `CTG/train/DG/parameter_analysis`, there is an analysis of two parameter adjustments: 
- Adjusting the number of retrieved sentences `k`.
- Analyzing the impact of SciQ training data volume on experimental results.

## Wikidata

Wikidata is too large to upload, so you need to rebuild the reverse index manually.

You can refer to the following guide:  
https://github.com/castorini/pyserini/blob/master/docs/usage-index.md#building-a-bm25-index-embeddable-python-implementation  

Specifically, follow the method under **Building a BM25 Index**.

I used the `enwiki-latest-pages-articles-multistream.xml.bz2` file from:  
https://dumps.wikimedia.org/enwiki/latest/  

to build the reverse index.
