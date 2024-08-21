# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os

from setuptools import find_packages, setup

# Package metadata
NAME = "evf_sam2"
VERSION = "1.0"
DESCRIPTION = "EVF SAM: Efficient Video Frame Segmentation with Segment Anything Model"
URL = "https://github.com/facebookresearch/segment-anything-2"
AUTHOR = "Meta AI"
AUTHOR_EMAIL = "segment-anything@meta.com"
LICENSE = "Apache 2.0"

# Read the contents of README file
with open("README.md", "r", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

# Required dependencies
REQUIRED_PACKAGES = [
    "packaging",
    "sentencepiece",
    "markdown2",
    "openai",
    "ray",
    "Requests",
    "shortuuid",
    "tqdm",
    "scipy",
    "bitsandbytes",
    "blobfile",
    "mypy",
    "pytest",
    "requests",
    "tensorboardX",
    "ftfy",
    "pyarrow",
    "torchmetrics==0.7.3",
    "deepspeed",
    "pycocoevalcap",
    "torchscale==0.2.0",
    "hydra-core",
]

# Setup configuration
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=URL,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    packages=find_packages(exclude=['tests', 'scripts']),
    include_package_data=True,
    install_requires=REQUIRED_PACKAGES,
    python_requires=">=3.10.0",
)