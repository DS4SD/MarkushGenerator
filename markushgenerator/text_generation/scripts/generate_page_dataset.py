#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# Set cache directories (Before importing transformers)
os.environ["HF_DATASETS_CACHE"] = "/mnt/volume/lum/.cache/huggingface/"
os.environ["HF_HOME"] = "/mnt/volume/lum/.cache/huggingface/"
os.environ["TRANSFORMERS_CACHE"] = "/mnt/volume/lum/.cache/transformers/"

import functools
import glob
import json
import multiprocessing
import shutil
from time import time

import datasets
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from rdkit import RDLogger
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")
from datasets import Dataset, concatenate_datasets

from markushgenerator.cxsmiles_tokenizer import CXSMILESTokenizer
from markushgenerator.text_generation.image_text_merging import ImageTextMerger
from markushgenerator.text_generation.text_augmentation import \
    DescriptionAugmentator
from markushgenerator.text_generation.text_generation import \
    DescriptionGenerator


def generate_samples_multiprocessing(
    dataset_cxsmiles_hf, dataset_size, page_dataset_name, do_augment, num_workers
):
    print("Calling generate_samples_multiprocessing")
    dataset_cxsmiles_hf_sub = dataset_cxsmiles_hf.select(
        [i for i in range(len(dataset_cxsmiles_hf)) if i < dataset_size]
    )
    dataset_cxsmiles_hf_splits = [
        dataset_cxsmiles_hf_sub.shard(num_shards=num_workers, index=i)
        for i in range(num_workers)
    ]

    pool = multiprocessing.Pool(num_workers)
    generate_samples_partial = functools.partial(
        generate_samples,
        dataset_size=dataset_size,
        page_dataset_name=page_dataset_name,
        do_augment=do_augment,
    )
    pool.map(generate_samples_partial, dataset_cxsmiles_hf_splits)
    pool.close()
    pool.join()


def generate_sample(
    id,
    mol,
    cxsmiles_dataset,
    cxsmiles,
    cxsmiles_opt,
    keypoints,
    image,
    cells,
    page_image_path,
    description_generator,
    image_text_merger,
    description_augmentation,
    do_augment,
    verbose=False,
):
    description, annotation = description_generator.generate(cxsmiles, cxsmiles_dataset)
    if do_augment:
        description = description_augmentation.augment(description, annotation)
    page, page_cells = image_text_merger.create_page(
        image, cells, description, display_cells=verbose
    )
    page.save(page_image_path)

    if verbose:
        draw = ImageDraw.Draw(page)
        for cell in page_cells:
            bbox = [p * page.size[0] for p in cell["bbox"]]
            draw.rectangle(
                ((bbox[0], bbox[1]), (bbox[2], bbox[3])), outline="green", width=5
            )
        plt.imshow(page)
        plt.savefig(f"test_{id}.png")
        plt.close()

    sample = {
        "id": id,
        "page_image_path": page_image_path,
        "description": description,
        "annotation": annotation,
        "mol": mol,
        "cxsmiles_dataset": cxsmiles_dataset,
        "cxsmiles": cxsmiles,
        "cxsmiles_opt": cxsmiles_opt,
        "keypoints": keypoints,
        "cells": page_cells,
    }
    return sample


def generate_samples(
    dataset_cxsmiles_hf, dataset_size, page_dataset_name, do_augment, verbose=False
):
    description_generator = DescriptionGenerator()
    image_text_merger = ImageTextMerger()
    description_augmentation = DescriptionAugmentator()

    for i, row in tqdm(
        enumerate(dataset_cxsmiles_hf),
        total=min(len(dataset_cxsmiles_hf), dataset_size),
    ):
        if i >= dataset_size:
            break

        page_image_path = (
            os.getcwd()
            + f"/../../data/dataset/{page_dataset_name}/page_images/{row['id']}.png"
        )

        sample = generate_sample(
            row["id"],
            row["mol"],
            row["cxsmiles_dataset"],
            row["cxsmiles"],
            row["cxsmiles_opt"],
            row["keypoints"],
            image,
            do_augment,
            row["cells"],
            page_image_path,
            description_generator,
            image_text_merger,
            description_augmentation,
            do_augment,
            verbose=verbose,
        )
        with open(
            os.path.dirname(__file__)
            + f"/../../data/dataset/{page_dataset_name}/samples/{row['id']}.json",
            "w",
        ) as json_file:
            json.dump(sample, json_file)


def read_samples(ids, max_i, page_dataset_name, ts):
    """
    Important Note: HuggingFace Dataset.from_generator() is wrongly implemented. It keeps in cache the result of read_samples(). It can not be disabled.
    To force a new run, it is needed to modify the input arguments.
    """
    print(f"Calling read_samples at {ts}")
    for i, id in tqdm(enumerate(ids), total=min(max_i, len(ids))):
        if i >= max_i:
            break
        if os.path.exists(
            os.path.dirname(__file__)
            + f"/../../data/dataset/{page_dataset_name}/samples/{id}.json"
        ):
            with open(
                os.path.dirname(__file__)
                + f"/../../data/dataset/{page_dataset_name}/samples/{id}.json",
                "r",
            ) as json_file:
                sample = json.load(json_file)
            sample["page_image"] = Image.open(sample["page_image_path"])
            yield sample


def main():
    """
    Note: "mdu_3008_aug" (20k), "mdu_3010_aug" (30k) and "mdu_3010_aug" (30k) are used in mdu_3012_aug.
    "mdu_3013_aug" (30k) is not used. "mdu_3014_aug" (30k) is not used.
    """
    cxsmiles_dataset_name = "ocxsr_3005"
    page_dataset_name = "mdu_3015_aug"

    num_processes_mp = 1  # 10
    dataset_size = 150000
    do_generate_samples = True
    clean = True
    do_read_samples = True
    skip_existing_samples = False
    do_augment = True

    if skip_existing_samples and clean:
        print("Incompatible arguments")
        exit(0)

    if (
        do_generate_samples
        and clean
        and os.path.exists(
            os.path.dirname(__file__) + f"/../../data/dataset/{page_dataset_name}/"
        )
    ):
        shutil.rmtree(
            os.path.dirname(__file__) + f"/../../data/dataset/{page_dataset_name}/"
        )

    if not (
        os.path.exists(
            os.path.dirname(__file__) + f"/../../data/dataset/{page_dataset_name}/"
        )
    ):
        os.mkdir(
            os.path.dirname(__file__) + f"/../../data/dataset/{page_dataset_name}/"
        )
        os.mkdir(
            os.path.dirname(__file__)
            + f"/../../data/dataset/{page_dataset_name}/samples/"
        )
        os.mkdir(
            os.path.dirname(__file__)
            + f"/../../data/dataset/{page_dataset_name}/page_images/"
        )

    # dataset_cxsmiles_hf = datasets.load_from_disk(os.path.dirname(__file__) + f"/../../../deepsearch-ai-unidoc/data/{cxsmiles_dataset_name}/", keep_in_memory=False)
    dataset_cxsmiles_hf = datasets.load_from_disk(
        os.path.dirname(__file__)
        + f"/../../../molecule-depictor-cdk/data/hf_dataset/{cxsmiles_dataset_name}/",
        keep_in_memory=False,
    )
    dataset_cxsmiles_hf = concatenate_datasets(
        (dataset_cxsmiles_hf["train"], dataset_cxsmiles_hf["test"])
    )
    dataset_cxsmiles_hf = dataset_cxsmiles_hf.shuffle()  # Since 28/10/24 only

    # Skip existing samples
    if skip_existing_samples:
        existing_ids = [
            int(p.split("/")[-1][:-5])
            for p in glob.glob(os.path.dirname(__file__) + f"/../../data/samples/*")
        ]
        dataset_ids = dataset_cxsmiles_hf["id"]
        dataset_cxsmiles_hf = dataset_cxsmiles_hf.select(
            [
                i
                for i in tqdm(range(len(dataset_cxsmiles_hf)))
                if not (dataset_ids[i] in existing_ids)
            ]
        )
        print(dataset_cxsmiles_hf)

    # Generate pages
    if do_generate_samples:
        generate_samples_multiprocessing(
            dataset_cxsmiles_hf,
            dataset_size,
            page_dataset_name,
            do_augment,
            num_processes_mp,
        )

    # Store in Hugging Face format
    if do_read_samples:
        f = functools.partial(
            read_samples,
            ids=dataset_cxsmiles_hf["id"],
            max_i=dataset_size,
            page_dataset_name=page_dataset_name,
            ts=f"{time()}",
        )
        dataset_pages_hf = Dataset.from_generator(f)
        dataset_pages_hf = dataset_pages_hf.train_test_split(test_size=0.1)
        dataset_pages_hf.save_to_disk(
            os.path.dirname(__file__)
            + f"/../../../deepsearch-ai-unidoc/data/{page_dataset_name}"
        )


if __name__ == "__main__":
    main()
