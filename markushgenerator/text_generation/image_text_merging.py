import glob
import os
import random
import re
from collections import defaultdict

import datasets
import yaml
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps

from markushgenerator.cxsmiles_tokenizer import CXSMILESTokenizer


class ImageTextMerger:
    def __init__(self):
        return

    def crop_image_and_update_bboxes(self, page_image, cells, bbox_crop):
        new_page_image = page_image.crop(bbox_crop)
        new_cells = []
        for cell in cells:
            new_cells.append(
                {
                    "text": cell["text"],
                    "bbox": [
                        (cell["bbox"][0] * page_image.width - bbox_crop[0])
                        / new_page_image.width,
                        (cell["bbox"][1] * page_image.height - bbox_crop[1])
                        / new_page_image.height,
                        (cell["bbox"][2] * page_image.width - bbox_crop[0])
                        / new_page_image.width,
                        (cell["bbox"][3] * page_image.height - bbox_crop[1])
                        / new_page_image.height,
                    ],
                }
            )
        return new_page_image, new_cells

    def square_with_white_borders_resize(
        self, image, bounding_boxes, output_page_width=None, output_page_height=None
    ):
        original_width, original_height = image.size
        scaling_factor = min(
            output_page_width / original_width, output_page_height / original_height
        )
        new_size = (
            int(original_width * scaling_factor),
            int(original_height * scaling_factor),
        )
        resized_image = image.resize(new_size, Image.LANCZOS)
        new_image = Image.new("RGB", (output_page_width, output_page_height), "white")
        top_left_x = (output_page_width - new_size[0]) // 2
        top_left_y = (output_page_height - new_size[1]) // 2
        new_image.paste(resized_image, (top_left_x, top_left_y))
        adjusted_bboxes = []
        for bbox in bounding_boxes:
            xmin, ymin, xmax, ymax = bbox
            xmin = int(xmin * scaling_factor) + top_left_x
            ymin = int(ymin * scaling_factor) + top_left_y
            xmax = int(xmax * scaling_factor) + top_left_x
            ymax = int(ymax * scaling_factor) + top_left_y
            adjusted_bboxes.append((xmin, ymin, xmax, ymax))
        return new_image, adjusted_bboxes

    def add_white_borders_and_resize(self, image, bounding_boxes, border_thickness=50):
        original_width, original_height = image.size
        bordered_image = ImageOps.expand(image, border=border_thickness, fill="white")
        bordered_width, bordered_height = bordered_image.size
        resized_image = bordered_image.resize(
            (original_width, original_height), Image.LANCZOS
        )
        scale_x = original_width / bordered_width
        scale_y = original_height / bordered_height
        adjusted_bboxes = []
        for bbox in bounding_boxes:
            xmin, ymin, xmax, ymax = bbox
            new_xmin = int((xmin + border_thickness) * scale_x)
            new_ymin = int((ymin + border_thickness) * scale_y)
            new_xmax = int((xmax + border_thickness) * scale_x)
            new_ymax = int((ymax + border_thickness) * scale_y)
            adjusted_bboxes.append((new_xmin, new_ymin, new_xmax, new_ymax))
        return resized_image, adjusted_bboxes

    def crop_resize_pad(
        self,
        page_image,
        page_cells,
        output_page_width=None,
        output_page_height=None,
        verbose=False,
    ):
        # Crop page
        # page_image_tmp = page_image.crop((1, 1, page_image.size[0] - 1, page_image.size[1] - 1))
        bbox_crop = ImageChops.invert(page_image.convert("RGB")).getbbox()
        if verbose:
            print(bbox_crop)
        page_image, page_cells = self.crop_image_and_update_bboxes(
            page_image, page_cells, bbox_crop
        )

        # Square the image by adding white borders
        bboxes = []
        for cell in page_cells:
            bbox = [
                int(cell["bbox"][0] * page_image.size[0]),
                int(cell["bbox"][1] * page_image.size[1]),
                int(cell["bbox"][2] * page_image.size[0]),
                int(cell["bbox"][3] * page_image.size[1]),
            ]
            bboxes.append(bbox)
        page_image, bboxes = self.square_with_white_borders_resize(
            page_image, bboxes, output_page_width, output_page_height
        )
        page_cells = [
            {
                "text": cell["text"],
                "bbox": [
                    box[0] / page_image.size[0],
                    box[1] / page_image.size[1],
                    box[2] / page_image.size[0],
                    box[3] / page_image.size[1],
                ],
            }
            for cell, box in zip(page_cells, bboxes)
        ]

        # Add borders
        bboxes = []
        for cell in page_cells:
            bbox = [
                int(cell["bbox"][0] * page_image.size[0]),
                int(cell["bbox"][1] * page_image.size[1]),
                int(cell["bbox"][2] * page_image.size[0]),
                int(cell["bbox"][3] * page_image.size[1]),
            ]
            bboxes.append(bbox)
        page_image, bboxes = self.add_white_borders_and_resize(page_image, bboxes)
        page_cells = [
            {
                "text": cell["text"],
                "bbox": [
                    box[0] / page_image.size[0],
                    box[1] / page_image.size[1],
                    box[2] / page_image.size[0],
                    box[3] / page_image.size[1],
                ],
            }
            for cell, box in zip(page_cells, bboxes)
        ]
        return page_image, page_cells

    def create_page(
        self, image, cells, text_description, display_cells=False, debug=False
    ):
        self.image = image
        self.cells = cells
        self.text_description = text_description
        self.text_description_lines = []
        self.text_image = None
        self.text_cells = []
        self.page_image = image
        self.page_cells = []

        # Define drawing parameters
        self.parameters = {
            "fontsize": int(random.uniform(30, 50)),
        }
        self.parameters.update(
            {
                "text_x_offset": 10,
                "text_y_offset": 10,
                "text_width": int(self.image.size[0] * 2),  # 1.5),
                "text_height": int(self.image.size[1] * 1.5),
                "text_line_spacing": int(random.uniform(40, 60)),
                "font": random.choice(
                    [
                        ImageFont.truetype(
                            f"{os.path.dirname(__file__)}/../../data/fonts/arial.ttf",
                            self.parameters["fontsize"],
                        ),
                        ImageFont.truetype(
                            f"{os.path.dirname(__file__)}/../../data/fonts/courier_new.ttf",
                            self.parameters["fontsize"],
                        ),
                        ImageFont.truetype(
                            f"{os.path.dirname(__file__)}/../../data/fonts/times.ttf",
                            self.parameters["fontsize"],
                        ),
                        ImageFont.truetype(
                            f"{os.path.dirname(__file__)}/../../data/fonts/verdana.ttf",
                            self.parameters["fontsize"],
                        ),
                        ImageFont.truetype(
                            f"{os.path.dirname(__file__)}/../../data/fonts/sans-serif.ttf",
                            self.parameters["fontsize"],
                        ),
                        ImageFont.truetype(
                            f"{os.path.dirname(__file__)}/../../data/fonts/lora.ttf",
                            self.parameters["fontsize"],
                        ),
                    ]
                ),
                "max_characters_per_line": 125
                - 1.5 * self.parameters["fontsize"]
                + int(
                    random.uniform(5, 25)
                ),  # 90 - 1.5*self.parameters["fontsize"] + int(random.uniform(10, 30)),
                "page_x_offset": 10,
                "text_x_offset_variable_range": [0, 20],
                "page_y_offset": 10,
                "text_image_spacing": int(random.uniform(5, 100)),
                "output_page_width": 1024,
                "output_page_height": 1024,
                "image_x_offset": int(random.uniform(10, 500)),
                "image_y_offset": 10,
            }
        )
        # Default
        # self.parameters = {
        #     "fontsize": 55,
        # }
        # self.parameters.update({
        #     "text_x_offset": 10,
        #     "text_y_offset": 10,
        #     "text_width": int(self.image.size[0]*2),
        #     "text_height": int(self.image.size[1]*1.5),
        #     "text_line_spacing": 60,
        #     "font": random.choice([
        #         ImageFont.truetype(f"{os.path.dirname(__file__)}/../data/fonts/arial.ttf", self.parameters["fontsize"]),
        #     ]),
        #     "max_characters_per_line": 125 - 1.5*self.parameters["fontsize"] + 15,
        #     "page_x_offset": 10,
        #     "text_x_offset_variable_range": [0, 20],
        #     "page_y_offset": 10,
        #     "text_image_spacing": 52,
        #     "output_page_width": 1024,
        #     "output_page_height": 1024,
        #     "image_x_offset": 255,
        #     "image_y_offset": 10
        # })

        self.parameters["page_width"] = (
            self.parameters["text_width"] + 2 * self.parameters["page_x_offset"]
        )
        self.parameters["page_height"] = (
            self.parameters["text_height"]
            + self.image.size[1]
            + self.parameters["text_image_spacing"]
            + 2 * self.parameters["page_y_offset"]
        )

        # Create text image
        self.create_text_image()

        # Merge text and molecule images
        self.merge(add_text_description=True)

        # Crop, resize and pad the image
        self.page_image, self.page_cells = self.crop_resize_pad(
            self.page_image,
            self.page_cells,
            self.parameters["output_page_width"],
            self.parameters["output_page_height"],
        )

        # Randomize cells ordering to avoid any bias
        random.shuffle(self.page_cells)

        if debug:
            from pprint import pprint

            pprint(self.page_cells)
            self.page_cells = [c for c in self.page_cells if c["bbox"][1] > 0.5]
        # Display cells
        if display_cells:
            # draw = ImageDraw.Draw(self.page_image)
            overlay = Image.new("RGBA", self.page_image.size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(overlay)
            for cell in self.page_cells:
                bbox = [
                    int(cell["bbox"][0] * self.page_image.size[0]),
                    int(cell["bbox"][1] * self.page_image.size[1]),
                    int(cell["bbox"][2] * self.page_image.size[0]),
                    int(cell["bbox"][3] * self.page_image.size[1]),
                ]
                draw.rectangle(bbox, (255, 0, 0, 50), width=7)

            self.page_image = Image.alpha_composite(
                self.page_image.convert("RGBA"), overlay
            )

        return self.page_image, self.page_cells

    def merge(self, add_text_description=True):
        self.page_image = Image.new(
            "RGB",
            (self.parameters["page_width"], self.parameters["page_height"]),
            (255, 255, 255),
        )

        # Paste
        self.page_image.paste(
            self.image,
            (self.parameters["image_x_offset"], self.parameters["image_y_offset"]),
        )

        if add_text_description:
            self.page_image.paste(
                self.text_image,
                (
                    self.parameters["page_x_offset"],
                    self.parameters["image_y_offset"]
                    + self.image.size[1]
                    + self.parameters["text_image_spacing"],
                ),
            )

        # Update Markush image cells
        for cell in self.cells:
            page_cell = {
                "bbox": [
                    (
                        cell["bbox"][0] * self.image.size[0]
                        + self.parameters["image_x_offset"]
                    )
                    / self.parameters["page_width"],
                    (
                        cell["bbox"][1] * self.image.size[1]
                        + self.parameters["image_y_offset"]
                    )
                    / self.parameters["page_height"],
                    (
                        cell["bbox"][2] * self.image.size[0]
                        + self.parameters["image_x_offset"]
                    )
                    / self.parameters["page_width"],
                    (
                        cell["bbox"][3] * self.image.size[1]
                        + self.parameters["image_y_offset"]
                    )
                    / self.parameters["page_height"],
                ],
                "text": cell["text"],
            }
            self.page_cells.append(page_cell)

        if add_text_description:
            # Update text description cells
            for cell in self.text_cells:
                page_cell = {
                    "bbox": [
                        (
                            cell["bbox"][0] * self.parameters["text_width"]
                            + self.parameters["page_x_offset"]
                        )
                        / self.parameters["page_width"],
                        (
                            cell["bbox"][1] * self.parameters["text_height"]
                            + self.parameters["image_y_offset"]
                            + self.image.size[1]
                            + self.parameters["text_image_spacing"]
                        )
                        / self.parameters["page_height"],
                        (
                            cell["bbox"][2] * self.parameters["text_width"]
                            + self.parameters["page_x_offset"]
                        )
                        / self.parameters["page_width"],
                        (
                            cell["bbox"][3] * self.parameters["text_height"]
                            + self.parameters["image_y_offset"]
                            + self.image.size[1]
                            + self.parameters["text_image_spacing"]
                        )
                        / self.parameters["page_height"],
                    ],
                    "text": cell["text"],
                }
                self.page_cells.append(page_cell)

    def clean_line(self, line):
        if line == "":
            return line
        if line[0] == " ":
            if len(line) > 1:
                line = line[1:]
            else:
                line = ""
        if len(line) > 0 and (line[-1] == " "):
            line = line[:-1]
        return line

    def create_text_image(self):
        # Split text description
        self.text_description_lines = []
        i = 0
        line = ""
        line_size = 0
        while i <= (len(self.text_description) - 1):
            line += self.text_description[i]
            line_size += 1
            if "\n" in self.text_description[i : i + 2]:
                self.text_description_lines.append(self.clean_line(line))
                line = ""
                line_size = 0
                i += 2
                continue
            if line_size >= self.parameters["max_characters_per_line"]:
                if (
                    (self.text_description[i] != " ")
                    and (self.text_description[i] != ",")
                    and (self.text_description[i] != ";")
                    and (self.text_description[i] != "-")
                ):
                    if (
                        ((i + 1) <= len(self.text_description) - 1)
                        and (self.text_description[i + 1] != " ")
                        and (self.text_description[i + 1] != " ")
                        and (self.text_description[i + 1] != ".")
                        and (self.text_description[i + 1] != ",")
                        and (self.text_description[i + 1] != ";")
                    ):
                        line += "-"
                self.text_description_lines.append(self.clean_line(line))
                line = ""
                line_size = 0
            i += 1

        self.text_description_lines.append(self.clean_line(line))

        # Display text description
        self.text_image = Image.new(
            "RGB",
            (self.parameters["text_width"], self.parameters["text_height"]),
            (255, 255, 255),
        )
        if self.text_description_lines == [""]:
            return
        draw = ImageDraw.Draw(self.text_image)
        current_y = self.parameters["text_y_offset"]
        for line in self.text_description_lines:
            if current_y > (
                self.parameters["text_height"] - self.parameters["text_y_offset"]
            ):
                break
            if current_y > (
                self.parameters["text_height"] - self.parameters["text_y_offset"]
            ):
                continue

            line_bbox = self.draw_text(draw, current_y, line)
            self.text_cells.append(
                {
                    "bbox": [
                        line_bbox[0] / self.parameters["text_width"],
                        line_bbox[1] / self.parameters["text_height"],
                        line_bbox[2] / self.parameters["text_width"],
                        line_bbox[3] / self.parameters["text_height"],
                    ],
                    "text": line,
                }
            )
            current_y += self.parameters["text_line_spacing"]

    def draw_text(self, draw, current_y, line):
        x = self.parameters["text_x_offset"] + random.uniform(
            *self.parameters["text_x_offset_variable_range"]
        )
        # Draw text
        draw.text((x, current_y), line, (0, 0, 0), font=self.parameters["font"])
        # Get bounding box
        line_size = self.parameters["font"].getsize(line)
        bbox = [x, current_y, x + line_size[0], current_y + line_size[1]]
        # Cap bounding box
        bbox = [
            max(bbox[0], 0),
            max(bbox[1], 0),
            min(bbox[2], self.parameters["text_width"]),
            min(bbox[3], self.parameters["text_height"]),
        ]
        return bbox
