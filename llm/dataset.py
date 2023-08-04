# dataset codes
# https://github.com/jeremyarancio/llm-tolkien/blob/main/llm/prepare_dataset.py

import json
from pathlib import Path
from typing import Callable, Mapping

from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer


def prepare_dataset(
    model_name: str,
    dataset_path: Path,
    min_length: int,
    context_length: int,
    test_size: float,
    shuffle: bool,
) -> None:
    """Prepare dataset for training and push it to the hub."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text = preprocess_data(
        dataset_path=dataset_path, min_length=min_length, tokenizer=tokenizer
    )
    dataset = Dataset.from_dict({"text": [text]})

    # tokenize dataset
    tokenized_datasets = dataset.map(
        tokenize,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "context_length": context_length},
        remove_columns=dataset.column_names,
    )
    tokenized_datasets.set_format("torch")

    tokenized_dataset_dict = tokenized_datasets.train_test_split(
        test_size=test_size, shuffle=shuffle
    )

    return tokenized_dataset_dict


def preprocess_data(
    dataset_path: Path, min_length: int, tokenizer: PreTrainedTokenizer
) -> str:
    """Prepare dataset for training from the jsonl file.

    Args:
        dataset_path (Path): Extracted text from the book
        min_length (int): Filter pages without text
        tokenizer (PreTrainedTokenizer): HuggingFace tokenizer

    Yields:
        str: text of the pages
    """
    with open(dataset_path, "r") as f:
        grouped_text = ""
        for line in f:
            elt = json.loads(line)
            text: str = list(elt.values())[0]
            if len(text) > min_length:
                grouped_text += text
        # End of paragraphs defined by ".\n is transformed into EOS token"
        grouped_text = grouped_text.replace(".\n", "." + tokenizer.eos_token)
        return preprocess_text(grouped_text)


def preprocess_text(text: str) -> str:
    text = text.replace("\n", " ")
    return text


def tokenize(element: Mapping, tokenizer: Callable, context_length: int) -> str:
    inputs = tokenizer(
        element["text"],
        truncation=True,
        return_overflowing_tokens=True,
        return_length=True,
        max_length=context_length,
    )
    inputs_batch = []
    for length, input_ids in zip(inputs["length"], inputs["input_ids"]):
        if (
            length == context_length
        ):  # We drop the last input_ids that are shorter than max_length
            inputs_batch.append(input_ids)
    return {"input_ids": inputs_batch}
