#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# Set cache directories (Before importing transformers)
os.environ["HF_DATASETS_CACHE"] = "/mnt/volume/lum/.cache/huggingface/"
os.environ["HF_HOME"] = "/mnt/volume/lum/.cache/huggingface/"
os.environ["TRANSFORMERS_CACHE"] = "/mnt/volume/lum/.cache/transformers/"

import re

import datasets
import torch
from accelerate import Accelerator
from datasets import concatenate_datasets, load_dataset
from huggingface_hub import login
from markushgrapher.core.common.markush_tokenizer import MarkushTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from markushgenerator.cxsmiles_tokenizer import CXSMILESTokenizer


class DescriptionAugmentator:
    def __init__(self):

        access_token = os.getenv("HF_TOKEN")
        login(token=access_token, new_session=True)

        model_id = "mistralai/Mistral-7B-Instruct-v0.3"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="cuda"
        )
        self.markush_tokenizer = MarkushTokenizer(tokenizer=None, dataset_path="")

    def augment(self, description, annotation, verbose=True):
        stable = self.markush_tokenizer.get_stable(annotation)
        if verbose:
            print("description:", description)
            print("stable:", stable)

        messages = [
            {
                "role": "user",
                "content": "I want you to augment a text description. Paraphrase it without changing its semantic meaning, but only its formulation. Do not add or remove any information. "
                "Use the writing style of patents in the chemistry domain. "
                "To help you preserving the semantic meaning of the description, a dictionnary is also provided. "
                "Its keys and values should not be modified in the augmented text description."
                "Directly answer with one augmented text description, and nothing else. Do not give any dictionnary output."
                f'Text description (to be paraphrased): "{description}". Dictionnary input (for context only): "{stable}".',
            }
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt"
        ).to("cuda")
        outputs = self.model.generate(input_ids, max_new_tokens=1000)
        augmented_description = self.tokenizer.decode(
            outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        augmented_description = augmented_description[
            len(messages[0]["content"]) :
        ].strip()
        if verbose:
            print(
                "------------------------------------------------------------------------------"
            )
            print("augmented_description:")
            print(augmented_description)
            print(
                "------------------------------------------------------------------------------"
            )

        # Note:
        # Postprocessing rules:
        # - descriptions starting with ' " '
        # - descriptions containing ' ictionnary '
        return augmented_description
