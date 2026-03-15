# OLR_to_Precipitation_Maps
Repository Structure
OLR_to_Precipitation_Maps/
│
├── dataset.py          # Dataset loader and preprocessing
├── models.py           # Generator and Discriminator architectures
├── train.py            # Training script for the cGAN model
├── utils.py            # Utility functions (checkpointing, logging, etc.)
├── config.py           # Configuration and hyperparameters
│
├── checkpoints/        # Saved model weights
│
├── data/               # Input datasets (not included in repository)
│
└── README.md
Environment Setup

It is recommended to run the project inside a Python virtual environment to manage dependencies.

macOS / Linux

Create a virtual environment:

python3 -m venv venv

Activate the environment:

source venv/bin/activate

Install required packages:

pip install -r requirements.txt

Deactivate the environment when finished:

deactivate
Windows

Create a virtual environment:

python -m venv venv

Activate the environment:

venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Deactivate when finished:

deactivate
Running the Project

After activating the environment, run the training script:

run the postprocess_test_only.py for results (set the argument --metric-max <threshold number in mm/day> to set a rainfall threshold)


#####
Large Files Notice
This repository uses Git Large File Storage (Git LFS) to manage large files such as model weights (.pth) and datasets.

If you download the repository as a ZIP from GitHub, you will only receive Git LFS pointer files instead of the actual large files.

To correctly obtain all required files, please clone the repository using Git with Git LFS installed:

git lfs install
git clone <repo-url>


If you have already cloned the repository but the large files appear as small text pointer files, run:

git lfs pull

If you cannot use Git LFS, you will need to manually download the large files from the repository and place them in the appropriate directories.
