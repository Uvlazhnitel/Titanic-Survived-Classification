# Titanic Classification Project

## Project Overview
This project aims to build a machine learning model to predict the survival of passengers aboard the Titanic based on various features.

**Task:** Binary classification of passenger survival (Survived).

**Business Use:** The model is intended for ranking/prioritization purposes. The focus is on achieving high recall while maintaining acceptable precision, with recall being the primary metric for optimization.

**Metrics:** PR-AUC (for handling class imbalance) and ROC-AUC. Later, confusion matrix, precision, recall, and F1-score will also be evaluated.

**Validation Protocol:** We will use StratifiedKFold(5) on the full training dataset (`train_full`). All preprocessing steps will be encapsulated within a `Pipeline` or `ColumnTransformer`. The test set will only be used once at the end.

## Dataset
The dataset used for this project is the Titanic dataset, which contains information about passengers such as age, gender, class, and survival status.

## Project Structure
- `data/`: Contains the dataset files.
- `notebooks/`: Jupyter notebooks for data exploration and model development.
- `src/`: Source code for data preprocessing, feature engineering, and model training.
- `README.md`: Project documentation.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/titanic-classification.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Run the data preprocessing script:
    ```bash
    python src/preprocess.py
    ```
2. Train the model:
    ```bash
    python src/train.py
    ```

## Results
Details about the model's performance and evaluation metrics will be added here.

