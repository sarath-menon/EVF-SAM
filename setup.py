import os

from setuptools import find_packages, setup

# Add this import at the top of the file
import site
import sys

site.ENABLE_USER_SITE = True

# Package metadata
NAME = "evf_sam"
VERSION = "1.0"
DESCRIPTION = "EVF-SAM: Efficient Video Frame-Level Segmentation with SAM"

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
    packages=find_packages(exclude=['tests', 'scripts']),
    include_package_data=True,
    install_requires=REQUIRED_PACKAGES,
    python_requires=">=3.10.0",
    # Add these new parameters
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
)

# Add this at the end of the file
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))