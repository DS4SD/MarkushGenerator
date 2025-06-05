#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import functools
import json
import multiprocessing
import os
import pathlib
import random
import shutil
import tempfile
from time import time

from cairosvg import svg2png
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import rdmolfiles
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

import datasets
import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets
from PIL import Image
from sklearn.utils import shuffle

from markushgenerator.cxsmiles_tokenizer import CXSMILESTokenizer
from markushgenerator.image_generation.generation import (
    generate_svg_image_process_star, get_boxes, get_cells)


def generate_samples_multiprocessing(
    dataset, dataset_name, max_i, use_generated_ids_only, num_workers
):
    dataset = dataset[:max_i]
    dataset_split = np.array_split(dataset, num_workers)
    pool = multiprocessing.Pool(num_workers)
    generate_samples_partial = functools.partial(
        generate_samples,
        dataset_name=dataset_name,
        max_i=max_i,
        use_generated_ids_only=use_generated_ids_only,
    )
    pool.map(generate_samples_partial, dataset_split)
    pool.close()
    pool.join()


def read_samples(dataset, dataset_name, max_i, ts):
    """
    Important Note: HuggingFace Dataset.from_generator() keeps in cache the result of read_samples().
    To force a new run, it is needed to modify input arguments and it is the role of the "ts" argument.
    """
    print(f"Calling read_samples at {ts}")
    for i, (idx, row) in tqdm(
        enumerate(dataset.iterrows()), total=min(max_i, len(dataset))
    ):
        if i >= max_i:
            break
        if os.path.exists(
            os.path.dirname(__file__)
            + f"/../data/dataset/{dataset_name}/samples/{row['id']}.json"
        ):
            with open(
                os.path.dirname(__file__)
                + f"/../data/dataset/{dataset_name}/samples/{row['id']}.json",
                "r",
            ) as json_file:
                data = json.load(json_file)

            data["image"] = Image.open(data["image_path"])
            yield data


def generate_sample(
    cxsmiles_dataset, id, svg_path, image_pil_path, molfile_path, cxsmiles_tokenizer
):
    # Replace subscripts unicode character in CXSMILES
    subscript_to_digit = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    cxsmiles_dataset = cxsmiles_dataset.translate(subscript_to_digit)

    # Replace subscripts unicode character in MOLFile
    with open(molfile_path, "r", encoding="utf-8") as file:
        content = file.read()
    with open(molfile_path, "w", encoding="utf-8") as file:
        file.write(content.translate(subscript_to_digit))

    # Convert SVG to PIL (30% of the time)
    image_file, image_filename_png = tempfile.mkstemp()
    svg2png(
        url=svg_path, write_to=image_filename_png, output_width=1024, output_height=1024
    )
    os.close(image_file)
    image_pil = Image.open(image_filename_png).convert("RGB")

    # Store PIL image
    image_pil.save(image_pil_path)

    # Convert MOL File to CXSMILES
    with rdBase.BlockLogs():
        molecule = Chem.MolFromMolFile(
            molfile_path, strictParsing=False, removeHs=False
        )  # sanitze = True
    if molecule is None:
        # print("Invalid CXSMILES from MOLfile")
        return None
    cxsmiles = Chem.MolToCXSmiles(molecule)

    # FIXME
    if molecule.HasProp("_smilesAtomOutputOrder"):
        mol_order = list(
            map(int, molecule.GetProp("_smilesAtomOutputOrder")[1:-2].split(","))
        )
        mol_to_cxsmi_i_mapping = {
            k: v for k, v in zip(mol_order, range(molecule.GetNumAtoms()))
        }
    else:
        # Fallback: assume identity mapping
        mol_to_cxsmi_i_mapping = {i: i for i in range(molecule.GetNumAtoms())}

    # Filter out CXSMILES with missing R labels
    original_r_labels = [
        c for c in cxsmiles_dataset.split("|")[1].split("$")[1].split(";") if c != ""
    ]
    if not (all([r in cxsmiles for r in original_r_labels])):
        # print("Invalid CXSMILES from MOLfile")
        return None

    # Get OCR cells  (60% of the time)
    atom_boxes, smt_boxes = get_boxes(svg_path)
    try:
        cells = get_cells(cxsmiles_dataset, molfile_path, atom_boxes, smt_boxes)
    except Exception as e:
        cells = None
    if cells == None:
        if os.path.exists(image_pil_path):
            os.remove(image_pil_path)
        if os.path.exists(svg_path):
            os.remove(svg_path)
            print(f"The generated image is removed.")
        return None

    # Replace subscripts unicode character in cells
    new_cells = []
    for cell in cells:
        new_cells.append(
            {"text": cell["text"].translate(subscript_to_digit), "bbox": cell["bbox"]}
        )
    cells = new_cells

    # Randomly shuffle cells to avoid giving any hint about the canonical SMILES ordering
    random.shuffle(cells)

    # Get optimized CXSMILES
    cxsmiles_opt, keypoints = cxsmiles_tokenizer.convert_cdk_to_opt(
        cxsmiles, molfile_path, mol_to_cxsmi_i_mapping
    )
    if cxsmiles_opt is None:
        print(f"Conversion to optimized CXSMILES failed.")
        return None

    # Get MOL block
    mol_block = rdmolfiles.MolToMolBlock(molecule, kekulize=False, forceV3000=True)
    if mol_block is None:
        print(f"Conversion to MOL block failed.")
        return None

    sample = {
        "id": id,
        "image_path": image_pil_path,
        "mol": mol_block,
        "cxsmiles": cxsmiles,
        "cxsmiles_dataset": cxsmiles_dataset,
        "cxsmiles_opt": cxsmiles_opt,
        "keypoints": keypoints,
        "cells": cells,
    }
    return sample


def generate_samples(dataset, dataset_name, max_i, use_generated_ids_only):
    if use_generated_ids_only:
        if max_i != len(dataset):
            print("Warning: 'max_i' and 'len(dataset)' are different")

    cxsmiles_tokenizer = CXSMILESTokenizer()
    for i, (idx, row) in tqdm(
        enumerate(dataset.iterrows()), total=min(max_i, len(dataset))
    ):
        if i >= max_i:
            break

        svg_path = (
            os.getcwd() + f"/../data/dataset/{dataset_name}/images/{row['id']}.svg"
        )
        if use_generated_ids_only and not (os.path.exists(svg_path)):
            continue
        molfile_path = (
            os.getcwd() + f"/../data/dataset/{dataset_name}/molfiles/{row['id']}.mol"
        )

        pathlib.Path(
            os.getcwd() + f"/../data/dataset/{dataset_name}/images_png/"
        ).mkdir(parents=True, exist_ok=True)
        image_pil_path = (
            os.getcwd() + f"/../data/dataset/{dataset_name}/images_png/{row['id']}.png"
        )

        sample = generate_sample(
            row["id"],
            row["cxsmiles"],
            svg_path,
            image_pil_path,
            molfile_path,
            cxsmiles_tokenizer,
        )
        if sample is None:
            continue
        with open(
            os.path.dirname(__file__)
            + f"/../data/dataset/{dataset_name}/samples/{row['id']}.json",
            "w",
        ) as json_file:
            json.dump(sample, json_file)


def main():
    experiment_name = "experiment-cx1000"
    # cxsmiles_dataset_path = os.path.dirname(__file__) + f"/../data/smiles/{experiment_name}.csv"
    base_ocsr_path = "/mnt/volume/lum/optical-chemical-structure-recognition/"
    cxsmiles_dataset_path = (
        base_ocsr_path + f"/data/pubchem/mixtures/raws/{experiment_name}.csv"
    )
    cxsmiles_only_dataset_path = (
        os.path.dirname(__file__)
        + f"/../data/smiles/{experiment_name}_cxsmiles_only.csv"
    )
    filter_cxsmiles_only = True
    dataset_name = "experiment-cx3000_cxsmiles_ocr"
    hf_dataset_name = "ocxsr_3000"
    hf_dataset_clean_name_1 = "ocxsr_3001"
    hf_dataset_clean_name_2 = "ocxsr_3002"
    clean = True
    generate_images = True  # Warning! Overwrites existing set with 'dataset_name'
    use_generated_ids_only = True
    num_processes_mp = 12
    clean_hf_dataset = True

    if filter_cxsmiles_only:
        print(f"Selecting lines in {cxsmiles_dataset_path} containing CXSMILES only")
        dataset = pd.read_csv(cxsmiles_dataset_path)
        dataset = dataset[dataset["cxsmiles"] == True]
        dataset_filtered = pd.DataFrame(
            {
                "id": range(len(dataset["isosmiles"])),
                "cxsmiles": dataset["isosmiles"],
            }
        )
        dataset_filtered.to_csv(cxsmiles_only_dataset_path)
        print(dataset_filtered)

    dataset = pd.read_csv(cxsmiles_only_dataset_path)
    dataset = shuffle(dataset)
    print(dataset)
    max_i = 300000

    if clean:
        if os.path.exists(
            os.path.dirname(__file__) + f"/../data/dataset/{dataset_name}"
        ):
            shutil.rmtree(
                os.path.dirname(__file__) + f"/../data/dataset/{dataset_name}"
            )

    if not (
        os.path.exists(os.path.dirname(__file__) + f"/../data/dataset/{dataset_name}")
    ):
        os.mkdir(os.path.dirname(__file__) + f"/../data/dataset/{dataset_name}")
        os.mkdir(os.path.dirname(__file__) + f"/../data/dataset/{dataset_name}/images")
        os.mkdir(
            os.path.dirname(__file__) + f"/../data/dataset/{dataset_name}/molfiles"
        )
        os.mkdir(os.path.dirname(__file__) + f"/../data/dataset/{dataset_name}/samples")

    # Generate images
    if generate_images:
        cxsmiles_dataset_list_split = np.array_split(
            list(dataset["cxsmiles"])[:max_i], num_processes_mp
        )
        ids_list_split = np.array_split(list(dataset["id"])[:max_i], num_processes_mp)
        args = [
            [
                cxsmiles_dataset_list_split[process_index],
                ids_list_split[process_index],
                dataset_name,
            ]
            for process_index in range(num_processes_mp)
        ]
        pool = multiprocessing.Pool(num_processes_mp)
        pool.map(generate_svg_image_process_star, args)
        pool.close()
        pool.join()

    # Generate bounding boxes and create dataset
    if num_processes_mp > 1:
        generate_samples_multiprocessing(
            dataset,
            dataset_name,
            max_i,
            use_generated_ids_only,
            num_workers=num_processes_mp,
        )
    else:
        generate_samples(
            dataset,
            dataset_name,
            max_i,
            use_generated_ids_only,
        )

    # Read generated samples and convert to hf
    dataset_hf = Dataset.from_generator(
        functools.partial(
            read_samples,
            dataset=dataset,
            dataset_name=dataset_name,
            max_i=max_i,
            ts=time(),
        )
    )

    dataset_hf = dataset_hf.train_test_split(test_size=0.1)
    dataset_hf.save_to_disk(
        os.path.dirname(__file__) + f"/../data/hf_dataset/{hf_dataset_name}/"
    )

    if clean_hf_dataset:
        # Remove CXSMILES opt conversion errors
        dataset_hf = concatenate_datasets(
            [
                datasets.load_from_disk(
                    os.path.dirname(__file__)
                    + f"/../data/hf_dataset/{hf_dataset_name}/",
                    keep_in_memory=False,
                )["train"],
                datasets.load_from_disk(
                    os.path.dirname(__file__)
                    + f"/../data/hf_dataset/{hf_dataset_name}/",
                    keep_in_memory=False,
                )["test"],
            ]
        )
        i_max = float("inf")
        verify = False
        remove_indices = []
        cxsmiles_tokenizer = CXSMILESTokenizer()
        for i, sample in tqdm(
            enumerate(dataset_hf.iter(batch_size=1)), total=min(i_max, len(dataset_hf))
        ):
            if i > i_max:
                break
            (
                id,
                image,
                mol,
                cxsmiles,
                cxsmiles_dataset,
                cxsmiles_opt,
                keypoints,
                cells,
            ) = (
                sample["id"][0],
                sample["image"][0],
                sample["mol"][0],
                sample["cxsmiles"][0],
                sample["cxsmiles_dataset"][0],
                sample["cxsmiles_opt"][0],
                sample["keypoints"][0],
                sample["cells"][0],
            )
            if "*" in cxsmiles_opt:
                remove_indices.append(i)
                continue
            if not (verify):
                continue
            try:
                cxsmiles_out = cxsmiles_tokenizer.convert_opt_to_out(cxsmiles_opt)
                molecule = Chem.MolFromSmiles(cxsmiles_out)
                if molecule is None:
                    print(cxsmiles_opt)
                    remove_indices.append(i)
                    continue
            except:
                break
        dataset_hf = dataset_hf.select(
            (i for i in range(len(dataset_hf)) if i not in set(remove_indices))
        )
        dataset_hf = dataset_hf.train_test_split(test_size=0.1)
        dataset_hf.save_to_disk(
            os.path.dirname(__file__)
            + f"/../data/hf_dataset/{hf_dataset_clean_name_1}/"
        )

        # Remove CXSMILES with multiple Sg sections on the same minimum or maximum atom indices
        dataset_hf = concatenate_datasets(
            [
                datasets.load_from_disk(
                    os.path.dirname(__file__)
                    + f"/../data/hf_dataset/{hf_dataset_clean_name_1}/",
                    keep_in_memory=False,
                )["train"],
                datasets.load_from_disk(
                    os.path.dirname(__file__)
                    + f"/../data/hf_dataset/{hf_dataset_clean_name_1}/",
                    keep_in_memory=False,
                )["test"],
            ]
        )
        remove_indices = []
        cxsmiles_tokenizer = CXSMILESTokenizer()
        for i, sample in tqdm(
            enumerate(dataset_hf.iter(batch_size=1)), total=min(i_max, len(dataset_hf))
        ):
            min_indices, max_indices = [], []
            for i_sample, section in enumerate(
                cxsmiles_tokenizer.parse_sections(
                    sample["cxsmiles_dataset"][0].split("|")[1]
                )
            ):
                if (len(section) >= 2) and (section[:2] == "Sg"):
                    sg_section = cxsmiles_tokenizer.parse_sg_section(section)
                    indices = []
                    for index in sg_section[2:]:
                        if index == "<atom_list_end>":
                            break
                        if index == ",":
                            continue
                        indices.append(int(index))
                    min_index, max_index = min(indices), max(indices)
                    if (min_index in min_indices) or (max_index in max_indices):
                        remove_indices.append(i)
                        break
                    min_indices.append(min_index)
                    max_indices.append(max_index)
        dataset_hf = dataset_hf.select(
            (i for i in range(len(dataset_hf)) if i not in set(remove_indices))
        )
        dataset_hf = dataset_hf.train_test_split(test_size=0.1)
        dataset_hf.save_to_disk(
            os.path.dirname(__file__)
            + f"/../data/hf_dataset/{hf_dataset_clean_name_2}/"
        )


if __name__ == "__main__":
    main()
