from pydantic import BaseSettings

class Settings(BaseSettings):
    # Model Settings
    MODEL_NAME: str = 't5-base' # t5-base or facebook/bart-base
    LEARNING_RATE: float = 1e-4 # 2e-5 for bart, 1e-4 for t5

    BATCH_SIZE: int = 12
    EPOCH_NUM: int = 4 #
    MAX_OUTPUT_LENGTH: int = 128
    MAX_INPUT_LENGTH: int = 512
    WEIGHT_DECAY: float = 0.0

    TRAIN_FILE: str = '/user_data/Cloze/dataset/sciq/mask_sentence_with_dtt/t5_new/parameter_analysis/sciq_all_mask_passage_with_dtt_for_t5_k=3.json'
    DEV_FILE: str = '/user_data/Cloze/dataset/sciq/mask_sentence_with_dtt/t5_new/parameter_analysis/sciq_all_mask_passage_with_dtt_for_t5_k=3.json'
    TEST_FILE: str = '/user_data/Cloze/dataset/sciq/mask_sentence_with_dtt/t5_new/parameter_analysis/sciq_all_mask_passage_with_dtt_for_t5_k=3.json'

    OUTPUT_DIR: str = '/user_data/Cloze/dtt_mask_lm_model/t5/parameter_analysis/sciq_all_passage_level_k=3' #/user_data/Cloze/dtt_mask_lm_model/t5/sciq_all_3dtt_passage_level_dtt_retrieve_12
    TASK_NAME: str = 'sciq_mlm'
    DATA_TYPE: str = 'sciq'
    WANDB_LOG_STEPS: int = 100


    GPUS: int = 2
    ACCELERATOR: str = 'ddp' #use dp pl 會不知道tensor 在哪個device

settings = Settings()