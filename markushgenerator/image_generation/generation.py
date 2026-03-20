#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pathlib
import re
import shlex
import subprocess
from tempfile import NamedTemporaryFile

import svgpathtools
from lxml import etree as ET
from rdkit import Chem
from svgpathtools import Line, Path, svgstr2paths, wsvg
from tqdm import tqdm

import numpy as np

def get_mol_elems(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    mol_elems = {"bond": {}, "atom": {}, None: {}}
    i = 0
    m = root.find("*/g[@class='mol']", namespaces=root.nsmap)
    for elem in m:
        svg_class = elem.get("class")
        if svg_class in ["bond", "atom", None]:
            i += 1
            id_ = elem.get("id")
            if id_ is None:
                id_ = i
            mol_elem = {
                "paths": [],
                "attribs": [],
            }
            for subelem in elem.iter():
                if len(subelem) == 0:
                    line = ET.tostring(subelem).decode().strip()
                    line = re.sub(
                        '^\s*(<)svg:(\w+)\s[\w\:="\/\.]+\s(.*)$', r"\1\2 \3", line
                    )
                    paths, attribs = svgstr2paths(line)
                    mol_elem["paths"].extend(paths)
                    mol_elem["attribs"].extend(attribs)
            mol_elems[svg_class][id_] = mol_elem
    return mol_elems


def paths_to_bbox(paths):
    xmin_paths, xmax_paths, ymin_paths, ymax_paths = paths[0].bbox()
    for path in paths:
        xmin_path, xmax_path, ymin_path, ymax_path = path.bbox()
        xmin_paths, xmax_paths, ymin_paths, ymax_paths = (
            min(xmin_paths, xmin_path),
            max(xmax_paths, xmax_path),
            min(ymin_paths, ymin_path),
            max(ymax_paths, ymax_path),
        )
    return [xmin_paths, ymin_paths, xmax_paths, ymax_paths]


def get_elem_bboxes(mol_elems):
    elem_bboxes = {"attribs": [], "paths": [], None: []}
    i = 0
    for type_, color in zip(["atom", "bond", None], ["green", "blue", "orange"]):
        for id_, obj in mol_elems.get(type_, {}).items():
            i += 1
            paths = obj.get("paths", [])
            bbox = paths_to_bbox(paths)
            attrib = {
                "id": f"mol1box{i}",
                "class": "bbox",
                "fill": "none",
                "stroke": color,
                "d": bbox.d(),
            }
            elem_bboxes["paths"].append(bbox)
            elem_bboxes["attribs"].append(attrib)
    edges = {}
    for id_, obj in mol_elems.get("bond", {}).items():
        paths = obj.get("paths", [])
        for path in paths:
            edges.setdefault(path.point(0), 0)
            edges[path.point(0)] += 1
            edges.setdefault(path.point(1), 0)
            edges[path.point(1)] += 1
    for edge, count in edges.items():
        if count > 1:
            i += 1
            edge = complex(edge)
            bbox = Path(
                Line(edge + (-2 - 2j), edge + (-2 + 2j)),
                Line(edge + (-2 + 2j), edge + (+2 + 2j)),
                Line(edge + (+2 + 2j), edge + (+2 - 2j)),
                Line(edge + (+2 - 2j), edge + (-2 - 2j)),
            )
            attrib = {
                "id": f"mol1cbox{i}",
                "class": "bbox",
                "fill": "none",
                "stroke": "red",
                "d": bbox.d(),
            }
            elem_bboxes["paths"].append(bbox)
            elem_bboxes["attribs"].append(attrib)
    return elem_bboxes


def merge_elem_bboxes(elem_bboxes, ifilename):
    paths = elem_bboxes.get("paths", [])
    attribs = elem_bboxes.get("attribs", [])
    tmpfile = NamedTemporaryFile(delete=False).name
    wsvg(paths, attributes=attribs, filename=tmpfile)
    with open(ifilename, "r") as fid1, open(tmpfile, "r") as fid2:
        data = fid1.read()
        m = re.search("^\s*<\/g>\s*<\/g>\s*</svg>", data, re.MULTILINE)
        s = m.span()
        new_data = data[: s[0]]
        for line in fid2:
            if re.search('^.*<path class="bbox"', line):
                new_data = new_data + line
        new_data = new_data + data[s[0] :]
    pathlib.Path(tmpfile).unlink(missing_ok=True)
    return new_data


def generate_svg_image(cxsmiles, id, dataset_name):
    """
    Compiles and runs the Depictor Java class.
    """
    current_dir = os.path.dirname(__file__)

    # Always compile Depictor.java
    java_compile_command = f'javac -cp "../../lib/*":. Depictor.java'
    compile_process = subprocess.Popen(
        shlex.split(java_compile_command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=current_dir,
    )
    compile_out, compile_err = compile_process.communicate(timeout=15)

    if compile_err:
        print("Compilation error:", compile_err.decode())
        return False

    # Run the Java class
    java_run_command = (
        f'java -cp "../../lib/*":. Depictor "{cxsmiles}" "{id}" "{dataset_name}"'
    )
    run_process = subprocess.Popen(
        shlex.split(java_run_command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=current_dir,
    )

    try:
        run_out, run_err = run_process.communicate(timeout=15)
    except subprocess.TimeoutExpired as e:
        print("Execution timed out:", e)
        return False

    if run_err:
        print("Java error:", run_err.decode())
        return False

    if run_out:
        print(run_out.decode())

    return True

def generate_svg_image_process(cxsmiles_dataset_list, ids_list, dataset_name):
    for cxsmiles, id in tqdm(
        zip(cxsmiles_dataset_list, ids_list), total=len(cxsmiles_dataset_list)
    ):
        success = generate_svg_image(cxsmiles, id, dataset_name)
        if not (success):
            print(f"Problem with id: {id}")


def generate_svg_image_process_star(cxsmiles_dataset_list_ids_list_dataset_name):
    return generate_svg_image_process(*cxsmiles_dataset_list_ids_list_dataset_name)


def get_boxes(svg_path, verbose=False):
    atoms_boxes, smt_boxes = {}, {}
    mol_elems = get_mol_elems(svg_path)
    counter = 0

    # Iterate over the molecule elements of 'None' type (neither atoms or bonds)
    # Parentheses are depicted using three elements. The label element is the last one.
    # Brackets are depicted using two elements. The first element has a stroke-width, and the second one is the label.
    for id_, obj in mol_elems.get(None, {}).items():
        paths, attribs = obj.get("paths", []), obj.get("attribs", [])
        if paths == [Path()]:
            # The label line of brackets can be null when the brackets have no labels
            # <path d='' stroke='none'/>
            counter = 0
            continue
        if verbose:
            print("Paths:", paths)
            print("Attribs:", attribs)

        # Other extra elements such as cycles (depicted with arcs) can also be flagged as None
        # They must be ignored when incrementing the counter to find brackets and parentheses labels
        # Some cases are not covered like this one:
        # <ellipse cx='70.41' cy='71.76' rx='49.06' ry='49.06' fill='none' stroke='#000000'/>
        # <path d='M220.57 162.6v12.39h-61.94v-12.39' fill='none' stroke='#000000' stroke-width='3.49'/>
        # <path d='M155.05 156.39l-10.73 -6.19l30.97 -53.64l10.73 6.19' fill='none' stroke='#000000' stroke-width='3.49'/>
        # <path d='M154.47 191.82l3.24 -4.63l-3.0 -4.26h1.89l1.35 2.08q.39 .6 .63 1.0q.35 -.55 .68 -.98l1.5 -2.1h1.79l-3.06 4.18l3.31 4.71h-1.85l-1.82 -2.76l-.48 -.74l-2.34 3.5h-1.82z' stroke='none'/>
        if any([isinstance(p, svgpathtools.path.Arc) for p in paths[0]]):
            continue

        counter += 1
        if "stroke-width" in attribs[0]:
            counter += 1

        if counter % 3 == 0:
            bbox = paths_to_bbox(paths)

            smt_boxes[counter // 3] = bbox

    for id_, obj in mol_elems.get("atom", {}).items():
        paths = obj.get("paths", [])
        bbox = paths_to_bbox(paths)
        atoms_boxes[id_] = bbox

    return atoms_boxes, smt_boxes


def get_cells(cxsmiles, molfile_path, atom_boxes, smt_boxes, verbose=False):
    if verbose:
        print(molfile_path)
    cells = []
    smt_texts = {}
    factor = 1 / 289

    # Read molecule from CXSMILES
    parser_params = Chem.SmilesParserParams()
    parser_params.allowCXSMILES = True
    parser_params.strictCXSMILES = False
    parser_params.removeHs = False
    molecule = Chem.MolFromSmiles(cxsmiles, parser_params)

    # Atoms
    for atom in molecule.GetAtoms():
        if (
            (atom.GetSymbol() == "C")
            and (atom.GetFormalCharge() == 0)
            and not (atom.HasProp("atomLabel"))
        ):
            continue
        if atom.HasProp("atomLabel"):
            # Rgroups
            text = atom.GetProp("atomLabel")
        else:
            text = atom.GetSymbol()

        if atom.GetFormalCharge() != 0:
            # For charges, it is important to be consistent everywhere: CDK, OCR, tokenizer.
            # Here, NH3+ -> N+1.
            #       Fe+3 -> Fe+3
            if "-" in str(atom.GetFormalCharge()):
                text += "-"
            else:
                text += "+"
            text += str(atom.GetFormalCharge()).replace("-", "").replace("+", "")
        if not (f"mol1atm{atom.GetIdx() + 1}" in atom_boxes):
            return None
        box = [p * factor for p in atom_boxes[f"mol1atm{atom.GetIdx() + 1}"]]

        cells.append({"bbox": [box[0], box[1], box[2], box[3]], "text": text})

    # SMT
    i = 1
    with open(molfile_path, "r") as f:
        for l in f.readlines():
            if not ("SRU" in l):  # V3000
                continue
            smt_line = [e for e in l.split(" ") if e != ""]
            if "CONNECT=HT" in smt_line:
                for field in smt_line:
                    if not ("LABEL" in field):
                        continue
                    smt_texts[i] = field[6:]
            else:
                smt_texts[i] = smt_line[8][6:]
            i += 1

    for i in range(1, len(smt_boxes) + 1):
        smt_box = [p * factor for p in smt_boxes[i]]
        cells.append(
            {
                "bbox": [smt_box[0], smt_box[1], smt_box[2], smt_box[3]],
                "text": smt_texts[i],
            }
        )

    return cells


def scale_bbox(x1, y1, x2, y2, scale_factor, img_width, img_height):
    width = x2 - x1
    height = y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    new_width = width * scale_factor
    new_height = height * scale_factor

    new_x1 = max(0, int(cx - new_width / 2))
    new_y1 = max(0, int(cy - new_height / 2))
    new_x2 = min(img_width, int(cx + new_width / 2))
    new_y2 = min(img_height, int(cy + new_height / 2))

    return new_x1, new_y1, new_x2, new_y2

def analyze_crop_edges(crop):

    np_img = np.array(crop.convert("L"))  # grayscale
    threshold = 128
    binary = np_img < threshold  # black pixels are True

    height, width = binary.shape

    # Edges
    top_edge = binary[0, :]
    bottom_edge = binary[-1, :]
    left_edge = binary[:, 0]
    right_edge = binary[:, -1]

    # Halves of top and bottom edges
    top_left_half = top_edge[:width // 2]
    top_right_half = top_edge[width // 2:]
    bottom_left_half = bottom_edge[:width // 2]
    bottom_right_half = bottom_edge[width // 2:]
    
    left_top_half = left_edge[:height // 2]
    left_bottom_half = left_edge[height // 2:]

    right_top_half = right_edge[:height // 2]
    right_bottom_half = right_edge[height // 2:]
    
    # Concatenate for left and right side pixels
    left_side_pixels = np.concatenate([top_left_half, bottom_left_half, left_edge])
    right_side_pixels = np.concatenate([top_right_half, bottom_right_half, right_edge])

    top_side_pixels = np.concatenate([left_top_half, right_top_half, top_edge])
    bottom_side_pixels = np.concatenate([left_bottom_half, right_bottom_half, bottom_edge])

    # Count black pixels
    left_black = int(np.sum(left_side_pixels))
    right_black = int(np.sum(right_side_pixels))
    top_black = int(np.sum(top_side_pixels))
    bottom_black = int(np.sum(bottom_side_pixels))

    if left_black==0 and right_black==0 and top_black==0 and bottom_black==0:
        return None 
    
    dominant_horizontal = "left" if left_black > right_black else "right"
    dominant_vertical = "top" if top_black > bottom_black else "bottom"

    return {
        "left_black": left_black,
        "right_black": right_black,
        "top_black": top_black,
        "bottom_black": bottom_black,
        "dominant_horizontal": dominant_horizontal,
        "dominant_vertical": dominant_vertical
    }


def process_image_with_bbox(pil_image, norm_bbox, scale_factor=2.0, step=0.1):
    
    img_width, img_height = pil_image.size

    # Convert normalized to absolute pixel coordinates
    x1 = int(norm_bbox[0] * img_width)
    y1 = int(norm_bbox[1] * img_height)
    x2 = int(norm_bbox[2] * img_width)
    y2 = int(norm_bbox[3] * img_height)

    analysis = None
    while not analysis:

        # Scale the absolute bbox
        sx1, sy1, sx2, sy2 = scale_bbox(x1, y1, x2, y2, scale_factor, img_width, img_height)

        # Crop and analyze
        crop = pil_image.crop((sx1, sy1, sx2, sy2))
        crop.save("crop.png")

        analysis = analyze_crop_edges(crop)
        if not analysis:
            scale_factor += step

    return {
        "normalized_bbox": norm_bbox,
        "absolute_bbox": (x1, y1, x2, y2),
        "scaled_bbox": (sx1, sy1, sx2, sy2),
        "analysis": analysis,
        "crop": crop  # Optional: remove if not needed
    }


def get_cells_v2(image_pil, cxsmiles, molfile_path, atom_boxes, smt_boxes, verbose=False):

    if verbose:
        print(molfile_path)
    cells = []
    smt_texts = {}
    factor = 1 / 289

    # Read molecule from CXSMILES
    parser_params = Chem.SmilesParserParams()
    parser_params.allowCXSMILES = True
    parser_params.strictCXSMILES = False
    parser_params.removeHs = False
    molecule = Chem.MolFromSmiles(cxsmiles, parser_params)
    
    num_atoms = molecule.GetNumAtoms(onlyExplicit=False) # Num of atoms in molecule  (including all Hs)
    num_atoms = molecule.GetNumAtoms(onlyExplicit=True) # Num of atoms in molecule  (only explicit ones)
    if verbose:
        print(num_atoms)

    # Atoms
    for atom in molecule.GetAtoms():
        neighbors = atom.GetNeighbors()
        if verbose:
            print(
                f"Atom: {atom.GetSymbol()}\n"
                f"  Degree: {atom.GetDegree()}\n"
                f"  Explicit Valence: {atom.GetExplicitValence()}\n"
                f"  Implicit Valence: {atom.GetImplicitValence()}\n"
                f"  Neighbors: {[neighbor.GetSymbol() for neighbor in neighbors]}\n"
                f"  Num of Implicit Hs: {atom.GetNumImplicitHs()}"
            )
    for atom in molecule.GetAtoms():
        if (
            (atom.GetSymbol() == "C")
            and (atom.GetFormalCharge() == 0)
            and not (atom.HasProp("atomLabel"))
            and not (f"mol1atm{atom.GetIdx() + 1}" in atom_boxes)
        ):  
            if verbose:
                print(f"C atom to reject: {atom.GetSymbol()}")
            continue
        if atom.HasProp("atomLabel"):
            # Rgroups
            text = atom.GetProp("atomLabel")
            if verbose:
                print(f"atomLabel: {text}")
        else:
            text = atom.GetSymbol()
            if verbose:
                print(f"normal Atom: {text}")

            if atom.GetSymbol() in ["O", "N", "S", "P", "F", "Cl", "Br", "I", "C"]:


                num_implicit_Hs = atom.GetNumImplicitHs() + atom.GetNumExplicitHs()

                if verbose:
                    print(f"Num of implicit Hs for {atom.GetSymbol()}: {atom.GetNumImplicitHs()}")

                bbox = [p * factor for p in atom_boxes[f"mol1atm{atom.GetIdx() + 1}"]]
                result_dict = process_image_with_bbox(image_pil, bbox, scale_factor=1.7, step=0.1) # og scale: 2
                if verbose:
                    print(result_dict) 

                dominant_horizontal = result_dict['analysis']['dominant_horizontal'] # left / right
                dominant_vertical = result_dict['analysis']['dominant_vertical'] # top / bottom
                    
                if num_implicit_Hs > 0: # if it has one or more implicit Hs
                        
                    x1, y1, x2, y2 = bbox
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1

                    if bbox_width >= bbox_height:
                        if verbose:
                            print("The bounding box is wider")
                        if dominant_horizontal == 'left':
                            if num_implicit_Hs == 1:
                                num_implicit_Hs = ""
                            text = text + "H"+ str(num_implicit_Hs)
                            if verbose:
                                print(text)
                        if dominant_horizontal == 'right':
                            if num_implicit_Hs == 1:
                                num_implicit_Hs = ""
                            text = "H"+ str(num_implicit_Hs) + text
                            if verbose:
                                print(text)

                    elif bbox_height > bbox_width:
                        if verbose:
                            print("The bounding box is taller")
                        if dominant_vertical == 'top':
                            if num_implicit_Hs == 1:
                                num_implicit_Hs = ""
                            text = text + "\\nH"+ str(num_implicit_Hs)
                            if verbose:
                                print(text)
                        if dominant_vertical == 'bottom':
                            if num_implicit_Hs == 1:
                                num_implicit_Hs = ""
                            text = "H" + str(num_implicit_Hs) + "\\n" + text
                            if verbose:
                                print(text)

        if atom.GetFormalCharge() != 0:
            # For charges, it is important to be consistent everywhere: CDK, OCR, tokenizer.
            # Here, NH3+ -> N+1.
            #       Fe+3 -> Fe+3
            if "-" in str(atom.GetFormalCharge()):
                text += "-"
            else:
                text += "+"
            text += str(atom.GetFormalCharge()).replace("-", "").replace("+", "")
        if not (f"mol1atm{atom.GetIdx() + 1}" in atom_boxes):
            continue
        box = [p * factor for p in atom_boxes[f"mol1atm{atom.GetIdx() + 1}"]]

        cells.append({"bbox": [box[0], box[1], box[2], box[3]], "text": text})

    # SMT
    i = 1
    with open(molfile_path, "r") as f:
        for l in f.readlines():
            if not ("SRU" in l):  # V3000
                continue
            smt_line = [e for e in l.split(" ") if e != ""]
            if "CONNECT=HT" in smt_line:
                for field in smt_line:
                    if not ("LABEL" in field):
                        continue
                    smt_texts[i] = field[6:]
            else:
                smt_texts[i] = smt_line[8][6:]
            i += 1

    for i in range(1, len(smt_boxes) + 1):
        smt_box = [p * factor for p in smt_boxes[i]]
        cells.append(
            {
                "bbox": [smt_box[0], smt_box[1], smt_box[2], smt_box[3]],
                "text": smt_texts[i],
            }
        )

    return cells