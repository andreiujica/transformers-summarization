import logging
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.utils import validate_pytorch2

def load_model_and_tokenizer(model_name: str) -> tuple:
    """
    load_model_and_tokenizer - load a model and tokenizer from a model name/ID on the hub
    :param str model_name: the model name/ID on the hub
    :return tuple: a tuple containing the model and tokenizer
    """
    logger = logging.getLogger(__name__)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
    ).to(device)
    model = model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info(f"Loaded model {model_name} to {device}")

    if validate_pytorch2():
        try:
            logger.info("Compiling model with Torch 2.0")
            model = torch.compile(model)
        except Exception as e:
            logger.warning(f"Could not compile model with Torch 2.0: {e}")
    else:
        logger.info("Torch 2.0 not detected, skipping compilation")

    return model, tokenizer



def summarize_and_score(ids, mask, model, tokenizer, **kwargs):
    """
    summarize_and_score - given a batch of ids and a mask, return a summary and a score for the summary
    Args:
        ids (): the batch of ids
        mask (): the attention mask for the batch
        model   (): the model to use for summarization
        tokenizer (): the tokenizer to use for summarization
    Returns:
        str: the summary of the batch
    """

    ids = ids[None, :]
    mask = mask[None, :]

    input_ids = ids.to("cuda") if torch.cuda.is_available() else ids
    attention_mask = mask.to("cuda") if torch.cuda.is_available() else mask

    model_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "return_dict_in_generate": True,
        "num_beams": 4,
        **kwargs
    }

    # Check if 'global_attention_mask' can be used
    if hasattr(model.config, 'attention_window'):
        global_attention_mask = torch.zeros_like(attention_mask)
        # Typically put global attention on the first token.
        global_attention_mask[:, 0] = 1
        model_kwargs['global_attention_mask'] = global_attention_mask

    summary_pred_ids = model.generate(**model_kwargs)

    summary = tokenizer.batch_decode(
        summary_pred_ids.sequences,
        skip_special_tokens=True,
        remove_invalid_values=True,
    )

    return summary


def summarize_via_tokenbatches(
    input_text: str,
    model,
    tokenizer,
    batch_length=2048,
    batch_stride=16,
    min_batch_length: int = 512,
    **kwargs,
):
    """
    summarize_via_tokenbatches - a function that takes a string and returns a summary
    Args:
        input_text (str): the text to summarize
        model (): the model to use for summarization
        tokenizer (): the tokenizer to use for summarization
        batch_length (int, optional): the length of each batch. Defaults to 2048.
        batch_stride (int, optional): the stride of each batch. Defaults to 16. The stride is the number of tokens that overlap between batches.
    Returns:
        str: the summary
    """
    # log all input parameters
    logger = logging.getLogger(__name__)
    # log all input parameters
    if batch_length < min_batch_length:
        logger.warning(
            f"batch_length must be at least {min_batch_length}. Setting batch_length to {min_batch_length}"
        )
        batch_length = min_batch_length

    logger.info(f"batch_length: {batch_length}, batch_stride: {batch_stride}")
    encoded_input = tokenizer.encode_plus(
        input_text, 
        add_special_tokens=True,
        max_length=batch_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_overflowing_tokens=True,
        stride=batch_stride
    )

    in_id_arr, att_arr = encoded_input.input_ids, encoded_input.attention_mask

    pbar = tqdm(total=len(in_id_arr), desc="Summarizing")

    if len(in_id_arr) <= 1:
        # If input fits in one batch, process it directly
        summary = summarize_and_score(in_id_arr[0], att_arr[0], model, tokenizer, **kwargs)
        pbar.update(len(in_id_arr))
    else:
        chunk_summaries = []
        for ids, mask in zip(in_id_arr, att_arr):
            summary = summarize_and_score(ids, mask, model, tokenizer, **kwargs)
            chunk_summaries.append(summary[0])
            pbar.update(1)
        
        # Concatenate chunk summaries and summarize again
        final_input_text = ' '.join(chunk_summaries)
        final_encoded_input = tokenizer.encode_plus(
            final_input_text, 
            add_special_tokens=True,
            max_length=batch_length,
            truncation=True,
            return_tensors="pt"
        )
        final_summary = summarize_and_score(final_encoded_input['input_ids'][0], final_encoded_input['attention_mask'][0], model, tokenizer, **kwargs)
        summary = final_summary

    pbar.close()
    return summary
