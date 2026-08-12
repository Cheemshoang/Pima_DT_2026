# Custom Tree-Based Models

This repository demonstrates the inner workings of Decision Trees, Random Forests, and XGBoost without relying on high-level machine learning libraries for the model architectures.

## Features & Implementations

- **Decision Tree (CART/ID3):** Supports both Gini Impurity and Information Gain (Entropy) for calculating optimal splits. Includes recursive tree-building and backtracking.
- **Random Forest:** Implements bootstrap aggregating (Bagging) and feature subsampling to construct an ensemble of independent decision trees, utilizing majority voting for final predictions.
- **XGBoost:** Features a gradient boosting implementation from scratch, utilizing residual learning, custom leaf weight calculations, and L2 regularization

## Project Structure

```text
├── data/
├── script/
│   ├── Decision_Tree.ipynb
│   ├── Random_Forest.ipynb
│   └── Xgboost.ipynb
├── src/
│   ├── dt.py
│   ├── rf.py
│   └── xgb.py
├── utils/
│   ├── loader.py
│   └── split_data.py
├── .gitignore
├── main.py
├── README.md
└── requirements.txt
```

## 🛠️ Setup & Installation

It is recommended to use a virtual environment to manage dependencies and isolate the project.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/wine_quality_project.git
cd wine_quality_project
```

### 2. Create and Activate a Virtual Environment

#### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation

First, split your raw dataset into training, validation, and test sets using the provided utility script. It will automatically separate the target variable (`quality`).

```bash
python data_split.py --input path/to/raw_data.csv --output ./data --y_column quality
```

### 2. Training and Evaluating Models

You can run the models using the main CLI script via the available arguments.

#### Run Decision Tree

```bash
python main.py dt --costfunc gini --max_depth 5 --min_split 3
```

#### Run Random Forest

```bash
python main.py rf --n_trees 10 --max_depth 10 --n_features 5
```

#### Run XGBoost

```bash
python main.py xgb --n_trees 50 --learning_rate 0.1 --max_depth 3 --lambda_reg 1.0
```
