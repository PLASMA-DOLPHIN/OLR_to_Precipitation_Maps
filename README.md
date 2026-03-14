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

Make sure the dataset files are placed inside the data/ directory before running the training script.
