import os
import torch
import pytorch_lightning as pl
from core.model import Model
from core.data_module import DataModule
from core.config import settings
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
from loguru import logger


if __name__ == "__main__":
    # init a model
    model = Model()
    # checkpoint = torch.load('/user_data/IWantToGraduate/squad_2_batch_24_lr_2e-05/checkpoints/epoch=01-dev_loss=1.13.ckpt')
    # model.load_state_dict(checkpoint['state_dict'])
    # DataModule
    dm = DataModule()

    # create output_dir
    if not (os.path.exists(settings.OUTPUT_DIR)):
        os.makedirs(settings.OUTPUT_DIR)
    

    # init wandb
    wandb_logger = WandbLogger(
        project=settings.TASK_NAME,
        save_dir=settings.OUTPUT_DIR,
        name="{0}_{1}".format(
            settings.MODEL_NAME,
            settings.DATA_TYPE
        ),
    )

    # trainer config
    trainer = pl.Trainer(
        gpus=settings.GPUS,
        accelerator=settings.ACCELERATOR,
        precision=32,
        default_root_dir=settings.OUTPUT_DIR,
        max_epochs=settings.EPOCH_NUM,
        logger=[wandb_logger],
        log_every_n_steps=settings.WANDB_LOG_STEPS,
        callbacks=[
            # EarlyStopping(monitor="dev_loss", patience=5, mode="min"),
            ModelCheckpoint(
                monitor="dev_loss",
                dirpath=os.path.join(settings.OUTPUT_DIR, "checkpoints"),
                filename="{epoch:02d}-{dev_loss:.2f}",
                save_top_k=settings.EPOCH_NUM,
                mode="min",
            ),
        ],
    )

    wandb_logger.watch(model)

    # train
    logger.info(f"Run Training!")
    # trainer.fit(model, datamodule=dm)
    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())

    wandb.finish()