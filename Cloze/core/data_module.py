from torch.utils.data import DataLoader, Dataset
import json
from transformers import DataCollatorWithPadding
from .tokenizer import get_tokenizer
import pytorch_lightning as pl
from .config import settings
import torch


class DataModule(pl.LightningDataModule):
    def __init__(self, tokenizer=get_tokenizer(settings.MODEL_NAME)):
        super().__init__()
        self.data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        self.batch_size = settings.BATCH_SIZE

        self.train_dataset= CLOTHADataset(split_set="train")
        self.dev_dataset = CLOTHADataset(split_set="dev")

    

    def train_dataloader(self):
        # 這邊如果回傳一個 dataloader, model 那邊會得到一個batch. 如果是兩個 dataloader, model 那邊會得到兩支 dataloader 各一個 batch 組成的 list
        # 這邊回傳兩個: 第一個是正樣本, 第二種是負樣本
        # print(self.train_dataset)
        return DataLoader(self.train_dataset, collate_fn=self.data_collator, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.dev_dataset, collate_fn=self.data_collator, batch_size=self.batch_size)

    # for predict
    def test_dataloader(self):
        return DataLoader(self.test_dataset, collate_fn=self.data_collator, batch_size=1, shuffle=False)


class DatasetUtilsMixin:
    def prepare_input(self, input, label):
        tokenizer = self.tokenizer
        tokenize_input = tokenizer(
            input,
            max_length=settings.MAX_INPUT_LENGTH,
            padding='max_length',
            truncation=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        label_input_ids = tokenizer(
            label,
            max_length=settings.MAX_OUTPUT_LENGTH,
            padding='max_length',
            truncation=True,
            add_special_tokens=True,
            return_tensors='pt'
        ).input_ids


        label_input_ids = label_input_ids.squeeze()
        for key in tokenize_input.keys():
            tokenize_input[key] = tokenize_input[key].squeeze()


        return {
            'input_ids':tokenize_input['input_ids'],
            'attention_mask':tokenize_input['attention_mask'],
            'label':label_input_ids
        }
    
class CLOTHADataset(Dataset, DatasetUtilsMixin):
    def __init__(
            self,
            split_set: str = "train",
            tokenizer=get_tokenizer(settings.MODEL_NAME),
            is_test=False,
        ):
        """
        Args:
            split_set(str): `train` or `validation`
            tokenizer(transformers.PreTrainedTokenizer)
        """
        if split_set == "train":
            with open(settings.TRAIN_FILE, "r", encoding="utf-8") as f_train:
                train_set = json.load(f_train)
                self.data = train_set
        elif split_set == "dev":
            with open(settings.DEV_FILE, "r", encoding="utf-8") as f_dev:
                dev_set = json.load(f_dev)
                self.data = dev_set
        elif split_set == "test":
            with open(settings.TEST_FILE, "r", encoding="utf-8") as f_test:
                test_set = json.load(f_test)
                self.data = test_set

        self.split_set = split_set
        self.is_test = is_test
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        data = self.data[index]

        sentence =  data["sentence"]
        label = data["label"]


        model_input = self.prepare_input(
            sentence,
            label
        )

        return model_input

    def __len__(self):
        return len(self.data)
    
