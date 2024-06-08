# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import json
import pathlib
from typing import Literal

import datasets
from datasets import load_dataset
from tqdm import tqdm

SplitType = Literal["train", "validation", "test"]


def argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["stanfordnlp/shp"],
        default="stanfordnlp/shp",
        help="A dataset to be downloaded.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        required=True,
        help="Output path for a downloaded dataset",
    )

    return parser.parse_args()


def create_prompt(input_data: str) -> str:
    """Create model formatted prompt from the data, adds control token if needed"""
    return input_data + "</s>\n"


def create_answer(input_data: str) -> str:
    """Create model formatted answer from the data, adds control token if needed"""
    return input_data


def convert_dataset_to_jsonl(
    dataset: datasets.Dataset,
    output_dir: pathlib.Path,
    split: SplitType,
):
    print(f"Converting {split} data...")
    with open(output_dir / f"{split}.jsonl", "w") as json_file:
        for sample in tqdm(dataset):
            prompt, answer_a, answer_b, selection = (
                create_prompt(sample["history"]),
                create_answer(sample["human_ref_A"]),
                create_answer(sample["human_ref_B"]),
                sample["labels"],
            )

            if selection == 1:
                confirmed, rejected = answer_a, answer_b
            else:
                confirmed, rejected = answer_b, answer_a

            data_line = json.dumps(
                {
                    "prompt": prompt,
                    "chosen_response": confirmed,
                    "rejected_response": rejected,
                }
            )
            json_file.write(f"{data_line}\n")


def main(args: argparse.Namespace) -> None:
    data = load_dataset(path=args.dataset)

    convert_dataset_to_jsonl(
        data["train"],
        output_dir=args.output,
        split="train",
    )
    convert_dataset_to_jsonl(
        data["validation"],
        output_dir=args.output,
        split="validation",
    )
    convert_dataset_to_jsonl(
        data["test"],
        output_dir=args.output,
        split="test",
    )


if __name__ == "__main__":
    args = argument_parser()
    main(args)
