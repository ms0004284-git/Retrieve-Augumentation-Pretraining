from transformers import T5Tokenizer, AutoTokenizer
from .config import settings


def get_tokenizer(model):
    if "tokenizer" not in globals():
        global tokenizer
        tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_NAME)

    return tokenizer


