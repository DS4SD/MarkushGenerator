#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ast
import re
import string

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdmolops


class CXSMILESTokenizer:
    def __init__(self, training_dataset=None, condense_labels=True):
        """
        Note:
        - Rlabels are tokenized using non-"others" tokens.
        - MOL files stores R labels as:
            - atoms (ex: K, U, ...)
            - atom properties (ex: atomProp:0.dummyLabel.J)
            - labels in $;$ section (ex: $;;R;;$)
        """
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
        self.atomic_symbols_exceptions = ["B", "Rb", "Re"]
        numbers_a = (
            [""]
            + [str(i) for i in list(range(0, 10))]
            + list(string.ascii_lowercase + string.ascii_uppercase)
        )
        numbers_b = (
            ["", "'"]
            + [str(i) for i in list(range(0, 10))]
            + list(string.ascii_lowercase + string.ascii_uppercase)
        )
        numbers_ab = [
            number_a + number_b for number_b in numbers_b for number_a in numbers_a
        ]
        numbers = ["", "'", "''"] + numbers_ab
        numbers = [
            n
            for n in numbers
            if (not (n in atomic_symbols)) or (n in self.atomic_symbols_exceptions)
        ]

        self.vocabulary = {
            "rlabel": [
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
                "T",
                "U",
                "V",
                "W",
                "X",
                "Y",
                "Z",
            ],
            "rlabel_number": [""]
            + [str(i) for i in list(range(0, 10))]
            + ["0" + str(i) for i in list(range(0, 10))],
            "rlabel_ring": [
                "A",
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
            "rlabel_ring_number": numbers,
            "smt": [c for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"],
        }
        self.vocabulary["all_r_label"] = list(
            np.unique(self.vocabulary["rlabel"] + self.vocabulary["rlabel_ring"])
        )
        self.training_dataset = training_dataset
        self.condense_labels = condense_labels

    def convert_cdk_to_opt(
        self, cxsmiles_mol, molfile_path, mol_to_cxsmi_i_mapping, verbose=False
    ):
        """
        1. Move R-groups labels from the extension table to the SMILES.
        2. Sort m sections. Warning: This is a test, it could be a bad idea since the current order may already follow the SMILES order.
        Note:
        - In CXSMILES, atom indices start from 0.
        - 'cxsmiles_mol' is used to read the molecule, 'molfile_path' is used to read m sections
        Warning:
        - In MOL file, R-group fragments must be "C[*]" and not "[*]C" for correct m sections reading.

        Important Note:
        - There is currently an error in both main training datasets.
        The indices are sometimes shifted. A good check would be to verify that cxsmiles_out and cxsmiles have markush equality = 1.
        """
        with rdBase.BlockLogs():
            parser_params = Chem.SmilesParserParams()
            parser_params.strictCXSMILES = False
            parser_params.sanitize = False
            parser_params.removeHs = False
            molecule = Chem.MolFromSmiles(cxsmiles_mol, parser_params)
        if molecule is None:
            print("Invalid CXSMILES")
            return None, []

        # Read rgroups
        # ... in atom properties
        rgroups = {}
        for i, atom in enumerate(molecule.GetAtoms()):
            if atom.HasProp("dummyLabel"):
                rgroups[i] = atom.GetProp("dummyLabel")
        # ... in $$ section (could may also be in atomLabel property)
        if "$" in cxsmiles_mol:
            for i, c in enumerate(cxsmiles_mol.split("$")[1].split(";")):
                if c == "":
                    continue
                rgroups[i] = c

        if verbose:
            print(f"R Groups: {rgroups}")

        # Read m groups from MOL File
        with rdBase.BlockLogs():
            molecule_mol = Chem.MolFromMolFile(
                molfile_path, strictParsing=False, removeHs=False, sanitize=False
            )
        m_section = {}
        for bond in molecule_mol.GetBonds():
            props = bond.GetPropsAsDict()
            if "_MolFileBondEndPts" in props:
                if bond.GetBeginAtom().GetSymbol() == "*":  # lum
                    atom_connector = bond.GetBeginAtom().GetIdx()
                elif bond.GetEndAtom().GetSymbol() == "*":
                    atom_connector = bond.GetEndAtom().GetIdx()
                else:
                    # Relies on the assumption that in GT molfiles, the fragments always start by the connecting atom
                    atom_connector = min(
                        bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()
                    )
                m_section[mol_to_cxsmi_i_mapping[atom_connector]] = [
                    str(mol_to_cxsmi_i_mapping[int(c) - 1])
                    for c in props["_MolFileBondEndPts"][1:-1].split(" ")[1:]
                ]

        # Sort m section atoms indices. Warning: This is a test, it could be a bad idea since the current order may already follow the SMILES order.
        m_section = {k: sorted(v, key=int) for k, v in m_section.items()}

        if verbose:
            print("m_section:", m_section)
        # Wildcard in smiles
        for i, rgroup in rgroups.items():
            molecule.GetAtomWithIdx(i).SetAtomicNum(0)
            molecule.GetAtomWithIdx(i).SetIsotope(i + 1)
        with rdBase.BlockLogs():
            cxsmiles = Chem.MolToCXSmiles(molecule, canonical=False)
        if verbose:
            print(f"cxsmiles: {cxsmiles}")

        smiles_opt = cxsmiles.split("|")[0].strip()
        if verbose:
            print(f"smiles_opt 1: {smiles_opt}")
        if self.condense_labels:
            for i, rgroup in rgroups.items():
                # Handle [1*@@H]
                first_i = smiles_opt.find(f"[{i+1}*")
                for i in range(first_i, len(smiles_opt)):
                    if smiles_opt[i] == "]":
                        last_i = i
                        break
                if self.training_dataset and ("mdu_2002" in self.training_dataset):
                    smiles_opt = smiles_opt.replace(
                        smiles_opt[first_i : last_i + 1], f"[{rgroup}]"
                    )
                else:
                    smiles_opt = smiles_opt.replace(
                        smiles_opt[first_i : last_i + 1], f"<r>{rgroup}</r>"
                    )

        # Replacement of exception atoms
        # The convention chosen here is that all these atoms are always considered as R-groups if they are in brackets (the boron B can be without brackets).
        # During training, they can be depicted as R-groups (without hydrogens).
        if self.condense_labels:
            for atom_symbol in self.atomic_symbols_exceptions:
                # Note: the [...@@H] case is not handled.
                smiles_opt = smiles_opt.replace(
                    f"[{atom_symbol}]", f"<r>{atom_symbol}</r>"
                )

        if verbose:
            print(f"smiles_opt 2: {smiles_opt}")

        if len(cxsmiles.split("|")) > 1:
            rtable = cxsmiles.split("|")[1]
        else:
            rtable = ""
        coordinates = rtable[rtable.find("(") : rtable.find(")") + 1]
        # Keypoints need to be rescaled
        keypoints = [
            ast.literal_eval("[" + c[:-1] + "]") for c in coordinates[1:-1].split(";")
        ]
        # Remove coordinates
        rtable = rtable.replace(coordinates, "")

        # Remove labels
        if self.condense_labels:
            rtable_opt = ""
        else:
            rtable_base = "$"
            for i, atom in enumerate(molecule.GetAtoms()):
                if i in rgroups:
                    rtable_base += rgroups[i]

                rtable_base += ";"
            rtable_base += "$,"
            rtable_opt = rtable_base
        rtable_split = rtable.split(",")

        for i, s in enumerate(rtable_split):
            if "atomProp" in s:
                continue
            if s == "":
                continue
            if not ("Sg" in s):
                continue

            # Consume Sg section
            if "Sg" in s:
                parsed_section = [c for c in s.split(":") if c != ""]
                s = ":".join(parsed_section)
                if len(parsed_section) == 3:
                    # Section contains comma and is splitted. (ex: Sg:n:3,4:H:ht) (ex: Sg:n:1,2,3:l:ht)
                    offset = 1
                    next_index = rtable_split[i + offset].split(":")

                    while len(next_index) == 1:
                        s += "," + next_index[0]
                        offset += 1
                        next_index = rtable_split[i + offset].split(":")
                        if verbose:
                            print("Next index:", next_index)

                    # Sg sections from cxsmiles mol can have empty fields (ex: Sg:n:9,10,11,12:X:ht:::)
                    parsed_section_end = [c for c in next_index if c != ""]
                    s += "," + ":".join(parsed_section_end)

            rtable_opt += s + ","

        # Add m section
        for idx, ring_indices in m_section.items():
            rtable_opt += "m:" + str(idx) + ":" + ".".join(ring_indices) + ","
        rtable_opt = rtable_opt[:-1]

        if verbose:
            print(f"rtable_opt: {rtable_opt}")
            print(f"rtable_split: {rtable_split}")

        if rtable_opt != "":
            cxsmiles_opt = smiles_opt + "|" + rtable_opt
        else:
            cxsmiles_opt = smiles_opt

        if verbose:
            print(f"cxsmiles_opt 3: {cxsmiles_opt}")
        return cxsmiles_opt, keypoints

    def parse_sections(self, rtable, split_on=","):
        if "$" in rtable:
            rtable = rtable[1:]  # Warning: To keep an eye on, not tested.
        sections = rtable.split(split_on)
        sections_list = []
        for i in range(len(sections)):
            if (len(sections[i]) >= 1) and (sections[i][0] == "m"):
                sections_list.append(sections[i])
            if (len(sections[i]) >= 2) and (sections[i][:2] == "Sg"):
                merged_section = sections[i] + ","
                j = i + 1
                while (
                    (j < len(sections))
                    and (len(sections[j]) >= 1)
                    and (sections[j][0] != "m")
                    and (sections[j][:2] != "Sg")
                ):
                    merged_section += sections[j] + ","
                    j += 1
                merged_section = merged_section[:-1]
                sections_list.append(merged_section)
        return sections_list

    def parse_m_section(self, section, split_on=":"):
        """
        Example: m:0:15.16.17.18.19.20
        """
        section_list = []
        if (len(section) >= 1) and (section[0] == "m"):
            m = section.split(split_on)[0]
            atom_connector = section.split(split_on)[1]
            atom_rings = section.split(split_on)[2].split(".")
            section_list.append(m)
            section_list.append(atom_connector)

            for atom_ring in atom_rings:
                section_list.append(atom_ring)
                section_list.append(".")
            section_list = section_list[:-1]
        return section_list

    def parse_sg_section(self, section, split_on=":"):
        """
        Example: Sg:n:11,12:F:ht
        """
        section_list = []
        if (len(section) >= 2) and (section[:2] == "Sg"):
            section_list.append(section.split(split_on)[0])
            section_list.append(section.split(split_on)[1])
            indices = section.split(split_on)[2].split(",")
            for index in indices:
                section_list.append(index)
                section_list.append(",")
            section_list = section_list[:-1]
            section_list.append("<atom_list_end>")
            section_list.extend(section.split(split_on)[3:])
        return section_list

    def get_rgroup_current_valence(self, atom_index, molecule):
        current_valence = 0
        for bond in molecule.GetAtomWithIdx(atom_index).GetBonds():
            bond_type = bond.GetBondType()
            # Convert aromatic bonds (which correspond to 13) for correct counting
            if bond_type == Chem.rdchem.BondType.AROMATIC:
                bond_type = 1.5
            if not (bond_type in [1, 1.5, 2, 3]):
                print(
                    f"Problem with get_rgroup_current_valence(), bond type is {bond_type}"
                )
            current_valence += bond_type
        return int(current_valence)

    def get_rgroup_type(self, rgroup_index, molecule, rgroup_valence):
        for fragment_indices in rdmolops.GetMolFrags(molecule):
            if (
                (rgroup_index in fragment_indices)
                and (len(fragment_indices) == 2)
                and (rgroup_valence == 1)
            ):
                return "residual"

        rings_atoms = molecule.GetRingInfo().AtomRings()
        for ring_atoms in rings_atoms:
            if rgroup_index in ring_atoms:
                return "heteroatom"

        if rgroup_valence > 1:
            return "linking"
        else:
            return "residual"

    def get_text_aliases(self, cxsmiles, cxsmiles_dataset, verbose=False):
        """
        Get aliases (R-groups and Sg identifier) in a CXSMILES.
        Note: By design of the CXSMILES, r-groups aliases are always before sg aliases in the list.
        """

        parser_params = Chem.SmilesParserParams()
        parser_params.strictCXSMILES = False
        parser_params.sanitize = False
        parser_params.removeHs = False
        molecule = Chem.MolFromSmiles(cxsmiles, parser_params)
        rgroups = {}

        for i, atom in enumerate(molecule.GetAtoms()):
            if atom.HasProp("dummyLabel"):
                rgroups[i] = atom.GetProp("dummyLabel")
        # ... in $$ section (could may also be in atomLabel property)
        if "$" in cxsmiles:
            for i, c in enumerate(cxsmiles.split("$")[1].split(";")):
                if c == "":
                    continue
                rgroups[i] = c

        aliases = []
        # Get R label
        for rgroup_index, rgroup_label in rgroups.items():
            valence = self.get_rgroup_current_valence(rgroup_index, molecule)
            aliases.append(
                {
                    "label": rgroup_label,
                    "type": "r",
                    "valence": valence,
                    "r_type": self.get_rgroup_type(rgroup_index, molecule, valence),
                }
            )

        # Get Sg label
        rtable = cxsmiles_dataset.split("|")[1]
        if verbose:
            print(rtable)
        for section in self.parse_sections(rtable):
            if verbose:
                print(section)
            if not (section[:2] == "Sg"):
                continue
            for i, item in enumerate(self.parse_sg_section(section)):
                if item == "<atom_list_end>":
                    label = self.parse_sg_section(section)[i + 1]
                    break
            aliases.append({"label": label, "type": "sg", "valence": 0})
        return aliases

    def convert_opt_to_out(
        self, cxsmiles_opt, from_prediction=True, verbose=False, r_group_separator="<r>"
    ):
        """
        Note:
        -   R groups connecting to cycles are not displayed in the output CXSMILES, but are correctly stored as 'atomLabel' property.
        """
        if verbose:
            print("cxsmiles_opt:", cxsmiles_opt)

        if cxsmiles_opt == None:
            return None
        cxsmiles = cxsmiles_opt.split("|")[0]

        i = 0
        rgroups = {}
        if self.condense_labels:
            if (self.training_dataset and ("mdu_2002" in self.training_dataset)) or (
                r_group_separator == "["
            ):
                for r_label in self.vocabulary["all_r_label"]:
                    for number in self.vocabulary["rlabel_ring_number"]:
                        r_label_number = r_label + number
                        if not ("[" + r_label_number + "]" in cxsmiles_opt):
                            continue
                        cxsmiles = cxsmiles.replace(
                            ("[" + r_label_number + "]"), f"[{i+1}*]"
                        )
                        rgroups[i] = r_label_number
                        i += 1
            else:
                for i, r_label_number in enumerate(
                    re.findall(r"<r>(.*?)<\/r>", cxsmiles_opt)
                ):
                    cxsmiles = cxsmiles.replace(
                        ("<r>" + r_label_number + "</r>"), f"[{i+1}*]"
                    )
                    rgroups[i] = r_label_number
                    i += 1

            if verbose:
                print("rgroups:", rgroups)

            rgroups_indices = {}
            molecule = Chem.MolFromSmiles(cxsmiles.split("|")[0], sanitize=False)
            if from_prediction and (molecule is None):
                print(
                    f"Can't convert optimized to out because the molecule is invalid: {cxsmiles}"
                )
                return None

            for i, atom in enumerate(molecule.GetAtoms()):
                if (atom.GetAtomicNum() == 0) and (atom.GetIsotope() != 0):
                    rgroups_indices[i] = rgroups[atom.GetIsotope() - 1]
            if verbose:
                print("rgroups_indices", rgroups_indices)

            # Add $$ section
            rtable = "$"
            for i, atom in enumerate(molecule.GetAtoms()):
                if i in rgroups_indices:
                    if (
                        self.training_dataset and ("mdu_2002" in self.training_dataset)
                    ) or (r_group_separator == "["):
                        rtable += rgroups_indices[i]
                    else:
                        rtable += rgroups_indices[i]
                rtable += ";"
            rtable += "$"
            cxsmiles += " |" + rtable
        else:
            cxsmiles += " |"
        # Append rest of rtable
        if len(cxsmiles_opt.split("|")) > 1:
            cxsmiles += "," + cxsmiles_opt.split("|")[1]
        cxsmiles += "|"
        if verbose:
            print("cxsmiles out:", cxsmiles)
        return cxsmiles
