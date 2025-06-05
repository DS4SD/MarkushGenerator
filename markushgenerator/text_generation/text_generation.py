import ast
import copy
import glob
import os
import random
import re
import string
from collections import defaultdict
from pprint import pprint

import datasets
import numpy as np
import pandas as pd
import yaml
from PIL import Image

from markushgenerator.cxsmiles_tokenizer import CXSMILESTokenizer


class DescriptionItem:
    def __init__(
        self,
        text_alias_group,
        templates,
        mappings,
        s_component_type_valence_s_components_mapping,
        parameters,
        item_index,
        ending_item,
        label_token,
        item_prefix_dl_template,
        item_dl_template,
        decoy_text_aliases,
    ):
        self.text_alias_group = text_alias_group
        self.templates = templates
        self.mappings = mappings
        self.s_component_type_valence_s_components_mapping = (
            s_component_type_valence_s_components_mapping
        )
        self.parameters = parameters
        self.item_index = item_index
        self.label_token = label_token  # r, r-list, sgv
        self.item_prefix_dl_template = item_prefix_dl_template
        self.item_dl_template = item_dl_template
        self.ending_item = ending_item
        self.decoy_text_aliases = decoy_text_aliases

        self.item_description = ""
        self.substituents_list = []

    def select_random_component(self, vocabulary, template_key):
        """
        Component can be "s-component-list", "int", "int-list", and also "r-list"
        """
        component = random.choice(vocabulary)
        if template_key == "s_component_atom_name":
            # Select random optional template
            self.s_component_atom_name_template = random.choice(
                self.templates["s_component_atom_name"]
            )

            # Randomly resolve template
            if (
                random.random()
                < self.parameters["optional_templates_application_proportions"][
                    "s_component_atom_name"
                ]
            ):
                component = self.s_component_atom_name_template["value"].replace(
                    "[s-component]", component
                )
        return component

    def create_list_from_vocabulary(
        self,
        nb_items,
        vocabulary,
        constraint="",
        max_duplicate_retries=5,
        template_key=None,
    ):
        list_items = []
        for _ in range(nb_items):
            if (constraint == "increasing") and (len(list_items) != 0):
                vocabulary = [
                    value
                    for value in vocabulary
                    if int(value) > max([int(item) for item in list_items])
                ]
                if vocabulary == []:
                    return list_items

            component = self.select_random_component(vocabulary, template_key)
            if constraint == "unique":
                if (len(vocabulary) > 1) and (component in vocabulary):
                    vocabulary.remove(component)
                else:
                    # s_component_atom_name optional template
                    nb_fails = 0
                    while (component in list_items) and (
                        nb_fails < max_duplicate_retries
                    ):
                        component = self.select_random_component(
                            vocabulary, template_key
                        )
                        nb_fails += 1

            list_items.append(component)
        return list_items

    def are_consecutive(self, lst):
        lst_sorted = sorted(lst)
        for i in range(len(lst_sorted) - 1):
            if lst_sorted[i + 1] - lst_sorted[i] != 1:
                return False
        return True

    def create_list_description(self, list_items, list_template, item_token):
        list = ""
        for item in list_items[:-1]:
            list += list_template["mid-text"].replace(item_token, item)
            list += list_template["separator"]
        list = list[: -len(list_template["separator"])]
        list += list_template["end-text"].replace(item_token, list_items[-1])
        return list

    def get_first_item_prefix(self):
        first_item_prefix = ""
        if random.random() < self.parameters["first_item_prefix_proportion"]:
            first_item_prefix = random.choice(
                list(self.mappings["first_item_prefix"].keys())
            )
            if (
                random.random()
                < self.parameters["optional_templates_application_proportions"][
                    "first_item_prefix"
                ]
            ):
                # Select random optional template
                self.first_item_prefix_template = random.choice(
                    self.templates["first_item_prefix"]
                )["value"]
                if "[f-id-s]" in self.first_item_prefix_template:
                    # Select random figure id
                    figure_id_s = random.choice(
                        list(self.mappings["figure_identifier"].keys())
                    )
                    figure_id_e = random.choice(
                        list(self.mappings["figure_identifier"].keys())
                    )
                    # Randomly resolve template
                    first_item_prefix = self.first_item_prefix_template.replace(
                        "[f-id-s]", figure_id_s
                    ).replace("[f-id-e]", figure_id_e)
                    # To avoid special characters like \n to be treated as literal characters
                    first_item_prefix = first_item_prefix.encode().decode(
                        "unicode_escape"
                    )
        return first_item_prefix

    def get_item_prefix_dl(self):
        # Replace [i]
        self.item_prefix_dl = self.item_prefix_dl_template["value"].replace(
            "[i]", str(self.item_index)
        )
        return self.item_prefix_dl

    def add_ending_filler(self):
        self.item_r_list_ending_filler_template = random.choice(
            self.templates["item_r_list_ending_filler"]
        )

        if self.label_token in self.item_r_list_ending_filler_template:
            self.item_description += self.item_r_list_ending_filler_template[
                "value"
            ].replace(self.label_token, self.r_list)

            # Replace multiple [s-component-list]
            while "[s-component-list]" in self.item_r_list_ending_filler_template:
                substituents_list_length = min(
                    random.choice(self.s_component_list_length_range),
                    len(s_components_subset),
                )
                s_component_type = np.random.choice(
                    list(self.s_component_types_available_proportions.keys()),
                    p=list(self.s_component_types_available_proportions.values()),
                )

                # s-component-list is created without any valence check
                valence = random.choice(
                    range(
                        max(
                            self.s_component_type_valence_s_components_mapping[
                                s_component_type
                            ]
                        )
                    )
                )
                s_components_subset = list(
                    self.s_component_type_valence_s_components_mapping[
                        s_component_type
                    ][valence].keys()
                )
                substituents_list = self.create_list_from_vocabulary(
                    substituents_list_length,
                    s_components_subset,
                    constraint=self.s_component_list_template["constraint"],
                    max_duplicate_retries=self.parameters[
                        "max_s_component_list_duplicate_retries"
                    ],
                    template_key=s_component_type,
                )
                s_component_list = self.create_list_description(
                    substituents_list,
                    self.s_component_list_template,
                    item_token="[s-component]",
                )

                self.item_description += self.item_r_list_ending_filler_template[
                    "value"
                ].replace("[s-component-list]", s_component_list, 1)

    def select_random_item_template(self):
        if self.item_dl_template != None:
            if self.label_token == "r-list":
                self.item_template = {
                    "value": self.item_dl_template["value"]
                    .replace("[label]", "[r-list]")
                    .replace("[value]", "[s-component-list]")
                    .encode()
                    .decode("unicode_escape")
                }
            if self.label_token == "r":
                self.item_template = {
                    "value": self.item_dl_template["value"]
                    .replace("[label]", "[r]")
                    .replace("[value]", "[s-component-list]")
                    .encode()
                    .decode("unicode_escape")
                }
            if self.label_token == "sgv":
                self.item_template = {
                    "value": self.item_dl_template["value"]
                    .replace("[label]", "[sgv]")
                    .replace("[value]", "[int-list]")
                    .encode()
                    .decode("unicode_escape")
                }
            return

        if self.label_token == "r-list":
            if (
                random.random()
                < self.parameters["optional_templates_application_proportions"][
                    f"item_{self.label_token.replace('-','_')}_filler"
                ]
            ):
                self.item_template = random.choice(
                    self.templates[f"item_{self.label_token.replace('-','_')}_filler"]
                )
                return
        self.item_template = random.choice(
            self.templates[f"item_{self.label_token.replace('-','_')}"]
        )

    def select_random_nested_template(self):
        if ("[s-component-list]" in self.item_template["value"]) or (
            "[s-component]" in self.item_template["value"]
        ):
            self.s_component_list_template = random.choice(
                self.templates["s_component_list"]
            )
        if "[r-list]" in self.item_template["value"]:
            self.r_list_template = random.choice(self.templates["r_list"])
        if "[int-list]" in self.item_template["value"]:
            self.int_list_template = random.choice(self.templates["int_list"])

    def generate_item_description(self):
        self.select_random_item_template()
        self.select_random_nested_template()

        if self.item_index == 0:
            self.item_description += self.get_first_item_prefix()

        if self.item_prefix_dl_template != None:
            self.item_description += self.get_item_prefix_dl()

        self.complete_template()

        if (len(self.text_alias_group) > 0) and (
            random.random()
            < self.parameters["optional_templates_application_proportions"][
                "item_r_list_ending_filler"
            ]
        ):
            self.add_ending_filler()

        if self.ending_item:
            if (
                random.random()
                < self.parameters["optional_templates_application_proportions"][
                    "last_item_ending_filler"
                ]
            ):
                last_item_ending_filler = random.choice(
                    list(self.mappings["last_item_ending_filler"].keys())
                )
                self.item_description += last_item_ending_filler

            if self.item_description[-len("\n") :] == "\n":
                self.item_description = self.item_description[: -len("\n")]

        return self.item_description

    def complete_template(self):
        if self.label_token == "r-list":
            self.complete_template_r_list()
        if self.label_token == "r":
            self.complete_template_r()
        if self.label_token == "sgv":
            self.complete_template_sgv()

        if "[s-component]" in self.item_template["value"]:
            self.complete_template_s_component()
        if "[s-component-list]" in self.item_template["value"]:
            self.complete_template_s_component_list()
        if "[int]" in self.item_template["value"]:
            self.complete_template_int()
        if "[int-list]" in self.item_template["value"]:
            self.complete_template_int_list()
        if "[int-s]" in self.item_template["value"]:
            self.complete_template_int_range()

    def complete_template_r_list(self):
        if "[r-list]" in self.item_template["value"]:
            # Create [r-list]
            self.r_list = self.create_list_from_vocabulary(
                len(self.text_alias_group),
                [text_alias["label"] for text_alias in self.text_alias_group],
                constraint=self.r_list_template["constraint"],
            )
            self.r_list = self.create_list_description(
                self.r_list, self.r_list_template, item_token="[r]"
            )

            # Replace [r-list]
            self.item_description += self.item_template["value"].replace(
                "[r-list]", self.r_list
            )

        if ("[r-s]" in self.item_template["value"]) and (
            "[r-e]" in self.item_template["value"]
        ):
            # Future TODO:
            # 1. Check that some CXSMILES in the dataset contain consecutive R1, R2, ...
            # 2. Implement "R1 to R4"
            return

    def complete_template_r(self):
        self.item_description += self.item_template["value"].replace(
            "[r]", self.text_alias_group[0]["label"]
        )

    def complete_template_sgv(self):
        self.item_description += self.item_template["value"].replace(
            "[sgv]", self.text_alias_group[0]["label"]
        )

    def get_s_component_list(self, substituents_list_length=None):
        # Select s-component type with at least one entry of valence self.text_alias["valence"]
        s_component_types_available_proportions = {
            type: proportion
            for type, proportion in zip(
                self.s_component_type_valence_s_components_mapping.keys(),
                list(self.parameters["s_component_proportions"].values()),
            )
            if (
                self.s_component_type_valence_s_components_mapping[type][
                    self.text_alias_group[0]["valence"]
                ]
                != {}
            )
        }

        # Renormalize proportions
        s_component_types_available_proportions = {
            t: p / sum(list(s_component_types_available_proportions.values()))
            for t, p in s_component_types_available_proportions.items()
        }

        # Draw random s-component type
        s_component_type = np.random.choice(
            list(s_component_types_available_proportions.keys()),
            p=list(s_component_types_available_proportions.values()),
        )
        s_components_subset = list(
            self.s_component_type_valence_s_components_mapping[s_component_type][
                self.text_alias_group[0]["valence"]
            ].keys()
        )

        # Select subtituents list length range
        if substituents_list_length == None:
            if random.random() < self.parameters["s_component_list_large_proportion"]:
                self.s_component_list_length_range = self.parameters[
                    "s_component_list_length_range_large"
                ]
            else:
                self.s_component_list_length_range = self.parameters[
                    "s_component_list_length_range"
                ]

            # Select substituents list length
            substituents_list_length = min(
                random.choice(self.s_component_list_length_range),
                len(s_components_subset),
            )

        # Replace [s-component-list]
        self.substituents_list = self.create_list_from_vocabulary(
            substituents_list_length,
            s_components_subset,
            constraint=self.s_component_list_template["constraint"],
            max_duplicate_retries=self.parameters[
                "max_s_component_list_duplicate_retries"
            ],
            template_key=s_component_type,
        )
        if random.random() < self.parameters["s_component_list_filler_ending"]:
            self.substituents_list.append(
                random.choice(list(self.mappings["s_component_filler"].keys()))
            )
        s_component_list = self.create_list_description(
            self.substituents_list,
            self.s_component_list_template,
            item_token="[s-component]",
        )
        return s_component_list

    def complete_template_s_component(self):
        _ = self.get_s_component_list(substituents_list_length=1)
        s_component_list = self.substituents_list[0]
        self.item_description = self.item_description.replace(
            "[s-component]", s_component_list
        )

    def complete_template_s_component_list(self):
        s_component_list = self.get_s_component_list()
        self.item_description = self.item_description.replace(
            "[s-component-list]", s_component_list
        )

    def complete_template_int(self):
        self.substituents_list = []
        int_values = []
        # The template uses multiple [int] occurences (for instance "[sgv] is [int] or [int]")
        int_value = random.choice(list(self.mappings["int"].keys()))
        while "[int]" in self.item_description:
            self.item_description = self.item_description.replace("[int]", int_value, 1)
            int_values.append(int(int_value))
            self.substituents_list.append(int_value)
            greater_int_values = [
                v for v in list(self.mappings["int"].keys()) if int(v) > max(int_values)
            ]
            if len(greater_int_values) > 0:
                int_value = random.choice(greater_int_values)
            else:
                int_value = random.choice(list(self.mappings["int"].keys()))

    def complete_template_int_list(self):
        self.substituents_list = self.create_list_from_vocabulary(
            random.choice(self.parameters["int_list_length_range"]),
            list(self.mappings["int"].keys()),
            constraint=self.int_list_template["constraint"],
        )
        int_list = self.create_list_description(
            self.substituents_list, self.int_list_template, item_token="[int]"
        )
        self.item_description = self.item_description.replace("[int-list]", int_list)

    def complete_template_int_range(self):
        self.substituents_list = []
        int_values = []
        int_e_presence = False
        if "[int-e]" in self.item_description:
            int_e_presence = True
        # The template uses multiple [int] occurences (for instance "[sgv] is [int] or [int]")
        int_value = random.choice(list(self.mappings["int"].keys()))
        for replace_string in ["[int-s]", "[int-e]"]:
            self.item_description = self.item_description.replace(
                replace_string, int_value, 1
            )
            int_values.append(int(int_value))
            self.substituents_list.append(int_value)
            self.substituents_list.append("-")
            if not (int_e_presence):
                return
            greater_int_values = [
                v for v in list(self.mappings["int"].keys()) if int(v) > max(int_values)
            ]
            if len(greater_int_values) > 0:
                int_value = random.choice(greater_int_values)
            else:
                int_value = random.choice(list(self.mappings["int"].keys()))
        self.substituents_list = self.substituents_list[:-1]

    def get_annotation(self, rtable_item_separator, substituents_separator):
        if self.label_token == "sgv":
            if "-" in self.substituents_list:
                if self.substituents_list[-1] != "-":
                    return (
                        f"{self.text_alias_group[0]['label']}:"
                        + self.substituents_list[0]
                        + "-"
                        + self.substituents_list[-1]
                    )
                else:
                    return (
                        f"{self.text_alias_group[0]['label']}:"
                        + self.substituents_list[0]
                        + "-"
                    )
            # If substituent list is a continous sequence of integers, simply represent it as "start index"-"end index"
            int_values = [int(s) for s in self.substituents_list]
            if self.are_consecutive(int_values):
                return (
                    f"{self.text_alias_group[0]['label']}:"
                    + str(min(int_values))
                    + "-"
                    + str(max(int_values))
                )
            return (
                f"{self.text_alias_group[0]['label']}:"
                + substituents_separator.join(self.substituents_list)
            )
        if (self.label_token == "r") or (self.label_token == "r-list"):
            # Remove fillers from substituents_list
            self.substituents_list = [
                s
                for s in self.substituents_list
                if not (s in list(self.mappings["s_component_filler"].keys()))
            ]
            if (self.label_token == "r-list") and (
                self.item_template in self.templates["item_r_list_filler"]
            ):
                self.substituents_list = []

        if self.label_token == "r":
            annotation = ""
            for text_alias in self.text_alias_group:
                if text_alias in self.decoy_text_aliases:
                    continue
                annotation += (
                    f"{text_alias['label']}:"
                    + substituents_separator.join(self.substituents_list)
                    + rtable_item_separator
                )
            if len(annotation) >= len(rtable_item_separator):
                annotation = annotation[: -len(rtable_item_separator)]
            return annotation
        if self.label_token == "r-list":
            text_alias_group_label = ""
            for text_alias in self.text_alias_group:
                if text_alias in self.decoy_text_aliases:
                    continue
                text_alias_group_label += text_alias["label"] + substituents_separator
            text_alias_group_label = text_alias_group_label[
                : -len(substituents_separator)
            ]

            annotation = (
                f"{text_alias_group_label}:"
                + substituents_separator.join(self.substituents_list)
                + rtable_item_separator
            )
            if len(annotation) >= len(rtable_item_separator):
                annotation = annotation[: -len(rtable_item_separator)]
            return annotation


class DescriptionGenerator:
    def __init__(self):
        """
        A description is constructed as:
            [item-label] is selected from [options] (given a random item template)
            [item-separator]
            ...
        Example: W5 is selected from aryl, heteroaryl, amido, or a methyl group; K is nitro, or propyl.
        """
        self.base_path_templates = (
            os.path.dirname(__file__) + "/../../data/text_templates/"
        )
        self.base_path_mappings = (
            os.path.dirname(__file__) + "/../../data/text_mappings/"
        )
        self.templates = self.get_templates()
        self.mappings = self.get_mappings()
        self.s_component_type_valence_s_components_mapping = (
            self.get_s_component_type_valence_s_components_mapping()
        )
        self.parameters = self.get_parameters()
        self.item_separator = ""
        self.rtable_item_separator = "<ns>"
        self.substituents_separator = "<n>"
        self.item_prefix_dl_template = None
        self.item_dl_template = None

    def select_random_dl_templates(self):
        if (
            random.random()
            < self.parameters["optional_templates_application_proportions"][
                "item_prefix_dl"
            ]
        ):
            self.item_prefix_dl_template = random.choice(
                self.templates["item_prefix_dl"]
            )
        else:
            self.item_prefix_dl_template = None
        if (
            random.random()
            < self.parameters["optional_templates_application_proportions"]["item_dl"]
        ):
            self.item_dl_template = random.choice(self.templates["item_dl"])
        else:
            self.item_dl_template = None

    def get_s_component_type_valence_s_components_mapping(self, max_valence=8):
        valence_s_components_lookups = {}
        for s_component_type, mapping_dict in self.mappings.items():
            if not ("s_component" in s_component_type):
                continue
            valence_s_components_lookups[s_component_type] = {}
            for valence in range(0, max_valence):
                d = {}
                for k, v in mapping_dict.items():
                    if isinstance(v, str):
                        v = ast.literal_eval(v)
                    if isinstance(v, int):
                        v = [v]
                    if v == None:
                        v = list(range(0, max_valence))
                    # The valence associated with a s-component can be a list of acceptable valences
                    if valence in v:
                        d[k] = v
                valence_s_components_lookups[s_component_type][valence] = d
        return valence_s_components_lookups

    def read_templates_list(self, relative_path):
        templates_list = []
        template_csv = pd.read_csv(self.base_path_templates + relative_path)
        for _, row in template_csv.iterrows():
            templates_list.append(row.to_dict())
        return templates_list

    def get_templates(self):
        templates = {}
        templates["r_list"] = self.read_templates_list("/list/r_list.csv")
        templates["s_component_atom_name"] = self.read_templates_list(
            "/s_component_atom_name.csv"
        )
        templates["s_component_list"] = self.read_templates_list(
            "/list/s_component_list.csv"
        )
        templates["int_list"] = self.read_templates_list("/list/int_list.csv")
        templates["item_r"] = self.read_templates_list("/item_r/item_r.csv")
        templates["item_r_list"] = self.read_templates_list("/item_r/item_r_list.csv")
        templates["item_sgv"] = self.read_templates_list("/item_sgv/item_sgv.csv")
        templates["item_dl"] = self.read_templates_list("/description/item_dl.csv")
        templates["item_prefix_dl"] = self.read_templates_list(
            "/description/item_prefix_dl.csv"
        )
        templates["first_item_prefix"] = self.read_templates_list(
            "/first_item_prefix.csv"
        )
        templates["r_list_fixed"] = self.read_templates_list("/list/fixed/r_list.csv")
        # Filler
        templates["item_r_list_filler"] = self.read_templates_list(
            "/item_r/filler/item_r_list_filler.csv"
        )
        templates["item_r_ending_filler"] = self.read_templates_list(
            "/item_r/filler/item_r_ending_filler.csv"
        )
        templates["item_r_list_ending_filler"] = self.read_templates_list(
            "/item_r/filler//item_r_list_ending_filler.csv"
        )
        return templates

    def read_mapping(self, relative_path):
        df = pd.read_csv(
            self.base_path_mappings + relative_path, dtype={"value": object}
        )
        values = [v.replace("\\n", "\n") for v in df["value"]]
        if "valence" in df.columns:
            return {k: v for k, v in zip(values, df["valence"])}
        else:
            return {k: None for k in values}

    def get_mappings(self):
        mappings = {}
        mappings["component_list_separator"] = self.read_mapping(
            "component_list_separator.csv"
        )
        mappings["int"] = self.read_mapping("int.csv")
        mappings["item_separator"] = self.read_mapping("item_separator.csv")
        mappings["s_component_manual_lum_test"] = self.read_mapping(
            "s_component_manual_lum_test.csv"
        )
        mappings["s_component_abbreviation_smiles"] = self.read_mapping(
            "s_component_abbreviation_smiles.csv"
        )
        mappings["s_component_abbreviation_name"] = self.read_mapping(
            "s_component_abbreviation_name.csv"
        )
        mappings["s_component_functional_group_smiles"] = self.read_mapping(
            "s_component_functional_group_smiles.csv"
        )
        mappings["s_component_functional_group_name"] = self.read_mapping(
            "s_component_functional_group_name.csv"
        )
        mappings["s_component_atom_smiles"] = self.read_mapping(
            "s_component_atom_smiles.csv"
        )
        mappings["s_component_atom_name"] = self.read_mapping(
            "s_component_atom_name.csv"
        )
        mappings["first_item_prefix"] = self.read_mapping("first_item_prefix.csv")
        mappings["figure_identifier"] = self.read_mapping("figure_identifier.csv")
        # Filler
        mappings["description_prefix_filler"] = self.read_mapping(
            "/filler/description_prefix_filler.csv"
        )
        mappings["s_component_filler"] = self.read_mapping(
            "/filler/s_component_filler.csv"
        )
        mappings["last_item_ending_filler"] = self.read_mapping(
            "last_item_ending_filler.csv"
        )
        return mappings

    def get_parameters(self):
        """Note: the order of keys in s_component_proportions must follow the same order than in self.mappings"""
        parameters = {
            # s-component-list
            "s_component_list_length_range": [2, 4],  # Minimum value is 2
            "s_component_list_length_range_large": [2, 10],
            "s_component_list_large_proportion": 0.3,  # 0.2
            "max_s_component_list_duplicate_retries": 5,
            "s_component_list_filler_ending": 0.05,
            # int-list
            "int_list_length_range": [2, 5],
            # r-list
            "r_list_proportion_dataset": 0.4,
            "r_list_proportion_sample": 0.8,
            "r_list_length_range": [2, 5],  # Minimum value is 2
            "r_list_number_range": [1, 3],
            # s-component
            "s_component_proportions": {
                "manual_lum_test": 0.85,  # 0.425,
                "abbreviation_smiles": 0,  # 0.02,
                "abbreviation_name": 0.05,  # 0.1,
                "functional_group_smiles": 0,  # 0.03,
                "functional_group_name": 0.025,  # 0.1,
                "s_component_atom_smiles": 0.025,  # 0.1,
                "s_component_atom_name": 0.05,  # 0.2,
            },
            # "s_component_proportions": {
            #     "manual_lum_test": 0.425,
            #     "abbreviation_smiles": 0.02,
            #     "abbreviation_name": 0.1,
            #     "functional_group_smiles": 0.03,
            #     "functional_group_name": 0.1,
            #     "s_component_atom_smiles": 0.1,
            #     "s_component_atom_name": 0.2,
            # },
            # Other
            "optional_templates_application_proportions": {
                "s_component_atom_name": 0.5,
                "first_item_prefix": 0.25,
                "item_r_list_filler": 0.025,
                "item_r_ending_filler": 0.05,
                "item_r_list_ending_filler": 0.05,
                "last_item_ending_filler": 0.025,
                "item_prefix_dl": 0.025,
                "item_dl": 0.025,
            },
            "first_item_prefix_proportion": 0.9,
            "image_only_proportion": 0.025,  # 0.1
            "decoy_text_alias_proportion": 0.05,
            "decoy_text_alias_length_range": [1, 3],
            "undefined_text_aliases_proportion_dataset": 0.025,
            "undefined_text_aliases_proportion_sample": 0.3,
        }
        return parameters

    def get_text_alias_groups(self, text_aliases):
        if random.random() < self.parameters["r_list_proportion_dataset"]:
            number_r_lists = random.choice(self.parameters["r_list_number_range"])
            r_lists_lengths = []
            for _ in range(number_r_lists):
                r_lists_lengths.append(
                    random.choice(self.parameters["r_list_length_range"])
                )

            text_aliases_for_rlist = []
            for text_alias in text_aliases:
                if (text_alias["type"] == "r") and (
                    random.random() < self.parameters["r_list_proportion_sample"]
                ):
                    text_aliases_for_rlist.append(text_alias)

            r_list_index = 0
            text_alias_groups = []
            for text_alias_1 in text_aliases_for_rlist:
                if any(
                    text_alias_1 in text_alias_group
                    for text_alias_group in text_alias_groups
                ):
                    continue
                text_alias_group = [text_alias_1]

                for text_alias_2 in text_aliases_for_rlist:
                    if len(text_alias_group) == r_lists_lengths[r_list_index]:
                        break
                    if text_alias_2 == text_alias_1:
                        continue
                    if any(
                        text_alias_1 in text_alias_group
                        for text_alias_group in text_alias_groups
                    ):
                        continue
                    if text_alias_1["valence"] != text_alias_2["valence"]:
                        continue
                    text_alias_group.append(text_alias_2)

                # Add groups with at least 2 text aliases in the group
                if len(text_alias_group) > 1:
                    text_alias_groups.append(text_alias_group)
                    r_list_index += 1

                if len(text_alias_groups) == number_r_lists:
                    break

            # Add everything that is not grouped yet as singletons
            for text_alias in text_aliases:
                if any(
                    [
                        text_alias in text_alias_group
                        for text_alias_group in text_alias_groups
                    ]
                ):
                    continue
                text_alias_groups.append([text_alias])

        else:
            text_alias_groups = [[g] for g in text_aliases]
        return text_alias_groups

    def create_rgroup_label(
        self, rlabel_vocabulary, is_quote_in_cxsmiles, is_subscript_in_cxsmiles
    ):
        extensions_proportions = {
            "display_rgroup_subscript": 0.2,
            "rgroup_index_nb": 0.3,
            "rgroup_index_nb_ab": 0.15,
        }
        rgroup_index_nb = list(string.digits)
        rgroup_index_quotes = ["'"] + ["''"]
        rgroup_index_nb_a = list(string.digits)
        rgroup_index_nb_b = list(string.digits)
        rgroup_index_character = list(
            string.ascii_lowercase
        )  # Note uppercase characters are not written as superscripts, so can not be easily covered here.
        atomic_symbols = [
            "H",
            "He",
            "Li",
            "Be",
            "B",
            "C",
            "N",
            "O",
            "F",
            "Ne",
            "Na",
            "Mg",
            "Al",
            "Si",
            "P",
            "S",
            "Cl",
            "Ar",
            "K",
            "Ca",
            "Sc",
            "Ti",
            "V",
            "Cr",
            "Mn",
            "Fe",
            "Co",
            "Ni",
            "Cu",
            "Zn",
            "Ga",
            "Ge",
            "As",
            "Se",
            "Br",
            "Kr",
            "Rb",
            "Sr",
            "Y",
            "Zr",
            "Nb",
            "Mo",
            "Tc",
            "Ru",
            "Rh",
            "Pd",
            "Ag",
            "Cd",
            "In",
            "Sn",
            "Sb",
            "Te",
            "I",
            "Xe",
            "Cs",
            "Ba",
            "La",
            "Ce",
            "Pr",
            "Nd",
            "Pm",
            "Sm",
            "Eu",
            "Gd",
            "Tb",
            "Dy",
            "Ho",
            "Er",
            "Tm",
            "Yb",
            "Lu",
            "Hf",
            "Ta",
            "W",
            "Re",
            "Os",
            "Ir",
            "Pt",
            "Au",
            "Hg",
            "Tl",
            "Pb",
            "Bi",
            "Po",
            "At",
            "Rn",
            "Fr",
            "Ra",
            "Ac",
            "Th",
            "Pa",
            "U",
            "Np",
            "Pu",
            "Am",
            "Cm",
            "Bk",
            "Cf",
            "Es",
            "Fm",
            "Md",
            "No",
            "Lr",
            "Rf",
            "Db",
            "Sg",
            "Bh",
            "Hs",
            "Mt",
            "Ds",
            "Rg",
            "Cn",
            "Nh",
            "Fl",
            "Mc",
            "Lv",
            "Ts",
            "Og",
        ]
        atomic_symbols_exceptions = ["B", "Rb", "Re"]
        subscript_to_digit = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

        random_character = random.choice(rlabel_vocabulary)
        random_number = ""
        random_sample = random.random()
        if 0 < random_sample < extensions_proportions["rgroup_index_nb"]:
            if is_subscript_in_cxsmiles:
                random_number = random.choice(rgroup_index_nb)
            else:
                random_number = random.choice(rgroup_index_nb + rgroup_index_quotes)
        elif (
            extensions_proportions["rgroup_index_nb"]
            < random_sample
            < (
                extensions_proportions["rgroup_index_nb"]
                + extensions_proportions["rgroup_index_nb_ab"]
            )
        ):
            random_number = random.choice(rgroup_index_nb_a) + random.choice(
                rgroup_index_nb_b
            )
        elif (
            (
                extensions_proportions["rgroup_index_nb"]
                + extensions_proportions["rgroup_index_nb_ab"]
            )
            < random_sample
            < (
                extensions_proportions["rgroup_index_nb"]
                + extensions_proportions["rgroup_index_nb_ab"]
                + extensions_proportions["rgroup_index_nb"]
            )
        ):
            random_character = "R"
            random_number = random.choice(rgroup_index_character)

        # Avoid rgroup labels to be atomic symbols
        if ((random_character + random_number) in atomic_symbols) and not (
            (random_character + random_number) in atomic_symbols_exceptions
        ):
            # Fallback option
            if is_subscript_in_cxsmiles:
                random_number = random.choice(rgroup_index_nb)
            else:
                random_number = random.choice(rgroup_index_nb + rgroup_index_quotes)
            return "R" + random_number

        # Display rgroup indices as subscripts (incompatible with quotes)
        if (
            not (is_quote_in_cxsmiles)
            and (random_number != "")
            and (random.random() < extensions_proportions["display_rgroup_subscript"])
        ):
            random_number = random_number.translate(subscript_to_digit)

        return random_character + random_number

    def generate(self, cxsmiles, cxsmiles_dataset):
        # Skip description generation
        if random.random() < self.parameters["image_only_proportion"]:
            return (
                "",
                "<markush>"
                + "<cxsmi>"
                + cxsmiles
                + "</cxsmi>"
                + "<stable>"
                + "</stable>"
                + "</markush>",
            )

        # Get description template
        self.select_random_dl_templates()

        # Random select item separator
        self.item_separator = random.choice(
            list(self.mappings["item_separator"].keys())
        )
        if self.item_dl_template != None:
            self.item_separator = self.item_dl_template["item_separator"]
        self.item_separator = self.item_separator.encode().decode("unicode_escape")

        # Get text aliases
        cxsmiles_tokenizer = CXSMILESTokenizer()
        text_aliases = cxsmiles_tokenizer.get_text_aliases(
            cxsmiles=cxsmiles, cxsmiles_dataset=cxsmiles_dataset
        )

        # Augment text aliases
        # Remove some text aliases which will be undefined
        if (
            random.random()
            < self.parameters["undefined_text_aliases_proportion_dataset"]
        ):
            text_aliases = [
                text_alias
                for text_alias in text_aliases
                if not (
                    random.random()
                    < self.parameters["undefined_text_aliases_proportion_sample"]
                )
            ]

        # Add decoy text aliaseses

        decoy_text_aliases = []
        if random.random() < self.parameters["decoy_text_alias_proportion"]:
            nb_decoy = random.choice(self.parameters["decoy_text_alias_length_range"])
            for _ in range(nb_decoy):
                random_rgroup_label = self.create_rgroup_label(
                    rlabel_vocabulary=[
                        "A",
                        "B",
                        "D",
                        "E",
                        "G",
                        "J",
                        "K",
                        "L",
                        "M",
                        "Q",
                        "R",
                        "T",
                        "U",
                        "V",
                        "W",
                        "X",
                        "Y",
                        "Z",
                    ],
                    is_quote_in_cxsmiles=False,
                    is_subscript_in_cxsmiles=False,
                )
                random_rgroup_label = random_rgroup_label.translate(
                    str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
                )
                decoy_text_aliases.append(
                    {
                        "label": random_rgroup_label,
                        "type": "r",
                        "valence": random.choice(range(1, 4)),
                        "r_type": "none",
                    }
                )
            text_aliases.extend(decoy_text_aliases)

        # Shuffle text_aliases
        random.shuffle(text_aliases)

        # Create text aliases groups (for R-lists)
        text_alias_groups = self.get_text_alias_groups(text_aliases)

        # Create description and annotation
        annotation = "<markush>" + "<cxsmi>" + cxsmiles + "</cxsmi>" + "<stable>"
        description = ""
        for i, text_alias_group in enumerate(text_alias_groups):
            if len(text_alias_group) == 0:
                continue

            if i == len(text_alias_groups):
                ending_item = True
            else:
                ending_item = False

            if len(text_alias_group) == 1:
                text_alias = text_alias_group[0]
                if text_alias["type"] == "sg":
                    label_token = "sgv"
                if text_alias["type"] == "r":
                    label_token = "r"

            if len(text_alias_group) > 1:
                label_token = "r-list"

            item = DescriptionItem(
                text_alias_group,
                self.templates,
                self.mappings,
                self.s_component_type_valence_s_components_mapping,
                self.parameters,
                i,
                ending_item,
                label_token,
                self.item_prefix_dl_template,
                self.item_dl_template,
                decoy_text_aliases,
            )

            # Generate description
            description += item.generate_item_description()
            description += self.item_separator

            # Generate annotation
            annotation = annotation + item.get_annotation(
                self.rtable_item_separator, self.substituents_separator
            )
            if (
                annotation[-len(self.rtable_item_separator) :]
                != self.rtable_item_separator
            ) and (annotation[-8:] != "<stable>"):
                annotation = annotation + self.rtable_item_separator

        if self.rtable_item_separator in annotation:
            annotation = annotation[
                : -len(self.rtable_item_separator)
            ]  # + "</stable>" + "</markush>"

        if description != "":
            description = description[: -len(self.item_separator)] + "."

        annotation = annotation + "</stable>" + "</markush>"
        return description, annotation

    def print_annotation(self, annotation):
        annotation = annotation.replace(
            self.rtable_item_separator, self.rtable_item_separator + "\n"
        )
        print(annotation)
