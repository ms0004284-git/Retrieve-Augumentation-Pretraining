import torch
import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration, T5Config, AutoModelForPreTraining, AutoConfig
from transformers import get_linear_schedule_with_warmup
from .tokenizer import get_tokenizer
from .config import settings
from .data_module import *
from loguru import logger



class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # self.save_hyperparameters()
        self.tokenizer = get_tokenizer(settings.MODEL_NAME)
        config = AutoConfig.from_pretrained(settings.MODEL_NAME)
        self.model = AutoModelForPreTraining.from_pretrained(settings.MODEL_NAME, config=config)
        self.model.resize_token_embeddings(len(self.tokenizer))


    def forward(self, **inputs):
        return self.model(**inputs, return_dict=True)
    
    def training_step(self, batch):
        outputs = self(**batch)
        loss = outputs.loss
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("dev_loss", loss.to(torch.float32), prog_bar=True)
        return loss
        
    def training_epoch_end(self, training_step_outputs):
        print('training_epoch_end')

    def configure_optimizers(self):
        dm = DataModule()
        model = self.model
        train_dataloader_size = len(dm.train_dataloader())
        # num_training_steps = (self.trainer.max_epochs * train_dataloader_size) / settings.BATCH_SIZE ##
        num_training_steps = self.trainer.max_epochs * train_dataloader_size
        num_warmup_steps = int(num_training_steps * 0.05)

        no_decay = ["bias", "LayerNorm.weight"]
        # ??
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": settings.WEIGHT_DECAY,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=settings.LEARNING_RATE,
            eps=1e-8,
        )

        self.lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        logger.info(
            f"optim scheduler is enable, num_warmup_steps:{num_warmup_steps} num_training_steps:{num_training_steps}"
        )
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()  # Step per iteration
        self.log("lr", self.lr_scheduler.get_last_lr()[0], prog_bar=True)

# for testing
    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        return outputs
    
    def test_epoch_end(self, outputs):
        with open('/user_data/Cloze/dataset/dmlm_keyword_output/', "w", encoding="utf-8") as f:
            json.dump(outputs, f)