from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from transformers import PreTrainedTokenizer
import torch

def sft_tulu_tokenize_and_truncate(row: Dict[str, Any], tokenizer: PreTrainedTokenizer, max_seq_length: int):
    """taken directly from https://github.com/allenai/open-instruct/blob/ba11286e5b9eb00d4ce5b40ef4cac1389888416a/open_instruct/finetune.py#L385"""
    
    messages = row["messages"]
    if len(messages) == 0:
        raise ValueError("messages field is empty.")
    input_ids = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=True,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_seq_length,
        add_generation_prompt=False,
    )
    labels = input_ids.clone()
    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            # we calculate the start index of this non-assistant message
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer.apply_chat_template(
                    conversation=messages[:message_idx],  # here marks the end of the previous messages
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=False,
                ).shape[1]
            # next, we calculate the end index of this non-assistant message
            if message_idx < len(messages) - 1 and messages[message_idx + 1]["role"] == "assistant":
                # for intermediate messages that follow with an assistant message, we need to
                # set `add_generation_prompt=True` to avoid the assistant generation prefix being included in the loss
                # (e.g., `<|assistant|>`)
                message_end_idx = tokenizer.apply_chat_template(
                    conversation=messages[: message_idx + 1],
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=True,
                ).shape[1]
            else:
                # for the last message or the message that doesn't follow with an assistant message,
                # we don't need to add the assistant generation prefix
                message_end_idx = tokenizer.apply_chat_template(
                    conversation=messages[: message_idx + 1],
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=False,
                ).shape[1]
            # set the label to -100 for the non-assistant part
            labels[:, message_start_idx:message_end_idx] = -100
            if max_seq_length and message_end_idx >= max_seq_length:
                break
    attention_mask = torch.ones_like(input_ids)
    
    return input_ids.flatten().tolist(), labels.flatten().tolist(), attention_mask.flatten().tolist()



def make_sft_collate(tokenizer: PreTrainedTokenizer,  max_seq_length: int, label_pad_token_id: int = -100):
    
    max_seq_length += 1  #to take account of the switch between input_ids and label
    
    def _apply(row):
        #tokenizer.padding_side = "left"
        return sft_tulu_tokenize_and_truncate(row, tokenizer, max_seq_length)

    def _pad_left(batch_seqs, pad_id, max_seq_length):
        #max_len = max(len(s) for s in batch_seqs)
        return [[pad_id] * (max_seq_length - len(s)) + s for s in batch_seqs]
    

    def _wrapped(batch):
        input_ids, attn, labels = [], [], []
        for row in batch:
        # batch est une liste de rows bruts (non tokenizés)
            ids, at, lbl = _apply(row)
            input_ids.append(ids)
            attn.append(at)
            labels.append(lbl)

        input_ids = _pad_left(input_ids, tokenizer.pad_token_id, max_seq_length)
        attn = _pad_left(attn, 0, max_seq_length)
        labels = _pad_left(labels, label_pad_token_id, max_seq_length)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attn = torch.tensor(attn, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        return input_ids[..., :-1], attn[..., :-1], labels[..., 1:]
    return _wrapped




