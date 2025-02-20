Getting started
===============

This guide will help you set up the project on a clean install.

## Prerequisites

Ensure you have the following installed:
- Python 3.12 or higher
- `pip` package manager

## Installation

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Set up environment variables:
    - Create a [.env](http://_vscodecontentref_/2) file in the root directory of the project.
    - Add any necessary environment variables to the [.env](http://_vscodecontentref_/3) file.

## Data Setup

1. Sync raw data:
    ```sh
    sync_data_from_s3 <s3-bucket-name> <local-directory>
    ```

2. Process the raw data to create the cleaned, final data sets:
    ```sh
    python scripts/process_data.py
    ```

## Directory Structure

The project directory should look like this:

├── .env ├── .gitignore ├── data │ ├── external <- Data from third party sources. │ ├── interim <- Intermediate data that has been transformed. │ ├── processed <- The final, canonical data sets for modeling. │ └── raw <- The original, immutable data dump. │ ├── docs <- Documentation files. │ ├── .gitkeep │ ├── docs │ ├── mkdocs.yml │ └── README.md │ ├── environment.yml <- Conda environment file. ├── LICENSE <- Open-source license. ├── models <- Trained and serialized models, model predictions, or model summaries. │ ├── .gitkeep │ └── autoencoder_model.pt │ ├── notebooks <- Jupyter notebooks. │ ├── .gitkeep │ ├── data_exploration │ ├── data_export │ └── data_processing │ ├── offshore_wind_nj <- Source code for use in this project. │ ├── init.py │ ├── pycache │ ├── config.py │ ├── convert_to_raster.py │ ├── convolutional_autoencoder.py │ ├── custom_image_dataset.py │ ├── data_cleaning.py │ ├── data_loader.py │ └── ... │ ├── pyproject.toml <- Project configuration file. ├── README.md <- The top-level README for developers using this project. ├── references <- Data dictionaries, manuals, and all other explanatory materials. │ ├── .gitkeep │ ├── reports <- Generated analysis as HTML, PDF, LaTeX, etc. │ └── figures <- Generated graphics and figures to be used in reporting. │ ├── requirements.txt <- The requirements file for reproducing the analysis environment. ├── setup.cfg <- Configuration file for flake8. └── Makefile <- Makefile with convenience commands like make data or make train.

## Running the Project

1. To run the project, use the following command:
    ```sh
    python main.py
    ```

2. To run Jupyter notebooks:
    ```sh
    jupyter notebook
    ```

## Additional Information

For more details, refer to the documentation.
