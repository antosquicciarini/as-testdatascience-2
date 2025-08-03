# Getting Started

This guide will help you set up and run the AS-TestDataScience-1 project locally.

## 🧰 Requirements

- Python 3.8 or later  
- Git  
- `make` (for UNIX systems) or Git Bash if using Windows  
- One of:  
  - [Conda](https://docs.conda.io/en/latest/miniconda.html) (recommended)  
  - Virtualenv / venv

## 🔄 Clone the Repository

    git clone https://github.com/antosquicciarini/as-testdatascience-2.git
    cd as-testdatascience-1

## 🧪 Create and Activate Environment

If you have Conda:

    make create_environment
    conda activate as-testdatascience-2

If using virtualenv instead:

    python3 -m venv .venv
    source .venv/bin/activate
    make requirements

## 📦 Install Dependencies

    make requirements

This will install all necessary Python packages from `requirements.txt`.

## ✅ Test the Environment

Check that Python version and packages are set up correctly:

    make test_environment

Expected output:

    >>> Development environment passes all tests!

## 🚀 Run the Notebook

Launch Jupyter and run the main notebook:

    jupyter notebook

Then open and execute:

    notebooks/energy_prediction.ipynb

## 📦 Extra Utilities

Or check code style:

    make lint