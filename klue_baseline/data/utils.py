# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import List, Optional, Union

import transformers
from transformers import PreTrainedTokenizer

from klue_baseline.data.base import InputExample, InputFeatures

logger = logging.getLogger(__name__)


def convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    label_list: List[str],
    max_length: Optional[int] = None,
    task_mode: Optional[str] = None,
) -> List[InputFeatures]:
    """Converts dataset in InputExample to dataset in InputFeatures to feed into pretrained models.

    This is for YNAT, KLUE-STS, KLUE-NLI, and KLUE-NER.

    Args:
        examples: List of InputExample converted from the raw dataset.
        tokenizer: Tokenizer of the pretrained model.
        label_list: List of labels of the task.
        max_length: Maximum length of the input tokens.
        task_mode: Task type.

    Returns:
        features: List of InputFeatures for the task and model.
    """
    if max_length is None:
        max_length = tokenizer.max_len

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample) -> Union[int, float, None, List[int]]:
        if example.label is None:
            return None
        if task_mode == "classification":
            return label_map[example.label]
        elif task_mode == "regression":
            return float(example.label)
        elif task_mode == "tagging":  # See KLUE paper: https://arxiv.org/pdf/2105.09680.pdf
            token_label = [label_map["O"]] * (max_length)
            for i, label in enumerate(example.label[: max_length - 2]):  # last [SEP] label -> 'O'
                token_label[i + 1] = label_map[label]  # first [CLS] label -> 'O'
            return token_label
        raise KeyError(task_mode)

    labels = [label_from_example(example) for example in examples]

    batch_encoding = tokenizer(
        [(example.text_a, example.text_b) for example in examples],
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i])

    return features


def check_tokenizer_type(tokenizer: PreTrainedTokenizer) -> str:
    """Checks tokenizer type.

    In KLUE paper, we only support wordpiece (BERT, KLUE-RoBERTa, ELECTRA) & sentencepiece (XLM-R).
    Will give warning if you use other tokenization. (e.g. bbpe)
    """
    if isinstance(tokenizer, transformers.XLMRobertaTokenizer):
        logger.info(f"Using {type(tokenizer).__name__} for fixing tokenization result")
        return "xlm-sp"  # Sentencepiece
    elif isinstance(tokenizer, transformers.BertTokenizer):
        logger.info(f"Using {type(tokenizer).__name__} for fixing tokenization result")
        return "bert-wp"  # Wordpiece (including BertTokenizer & ElectraTokenizer)
    else:
        logger.warn(
            "If you are using other tokenizer (e.g. bbpe), you should change code in `fix_tokenization_error()`"
        )
        return "other"
