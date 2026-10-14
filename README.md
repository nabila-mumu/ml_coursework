# Machine Learning Coursework

Practical implementations of **data preprocessing, exploratory data analysis, neural-network classification, and tree-based ensemble learning** developed as part of CSE 472 machine learning coursework. The repository includes Python notebooks for working with real datasets and **from-scratch implementations of Decision Trees, Random Forests, and Extra Trees**, compared against scikit-learn baselines.

## 1. Data preprocessing and diabetes classification

**Notebook:** [`2005050.ipynb`](data_preprocessing_offline/2005050/2005050.ipynb)  
**Dataset:** `medical_students_dataset.csv`  
**Target:** `Diabetes`

The notebook builds a complete experimental workflow:

1. **Dataset exploration:** Inspect shape and descriptive statistics; visualize age, BMI, and gender; count missing values and duplicate rows.
2. **Data cleaning:** Drop samples with missing `Diabetes` labels, fill missing numeric values with each column's mean, fill `Gender`, `Blood Type`, and `Smoking` using their modes, and remove duplicate records.
3. **Feature preparation:** Separate predictors from the target; use `LabelEncoder` for the target, gender, and smoking; and one-hot encode blood type with `pandas.get_dummies`.
4. **Scaling:** Define a reusable feature-scaling function supporting `StandardScaler` or `MinMaxScaler`. The notebook runs it with **StandardScaler** for selected numeric columns, leaving binary/one-hot columns unscaled.
5. **Correlation and feature selection:** Calculate feature–target correlations; visualize a correlation heatmap and scatter plots; select the **10 features with highest absolute correlation** with the diabetes label.
6. **Dataset splitting:** Perform stratified **70% training / 15% validation / 15% testing** splits with a fixed random seed.
7. **Neural-network training:** Convert data to PyTorch tensors and `Dataset`/`DataLoader` batches; train and compare three feed-forward architectures:
   - `FNN1`: two hidden layers (128, 64 units)
   - `FNN2`: three hidden layers (256, 128, 64 units)
   - `FNN3` (`FNN-wide`): two hidden layers (200, 100 units)
8. **Model selection and evaluation:** Train with **CrossEntropyLoss**, **Adam**, and ReLU hidden activations; compare validation performance; evaluate the selected network using **accuracy, macro precision, macro F1, and AUROC**; plot an ROC curve, confusion matrix output, and training-versus-validation loss comparison.

The notebook uses a learning rate of **0.001** and a default training duration of **20 epochs** in its training function. It automatically selects CUDA when available, otherwise CPU.

## 2. Tree-based classifiers implemented from scratch

**Entry point:** [`decision_tree_offline/main.py`](decision_tree_offline/main.py)

The tree models use **NumPy-based custom implementations**, while scikit-learn is used for datasets, splits, metrics, and benchmark models.

### Custom Decision Tree — `decision_tree.py`

- Defines a `TreeNode` storing node impurity, sample counts, class counts, split feature, threshold, and children.
- Calculates **Gini impurity**: \(G = 1 - \sum_k p_k^2\).
- Generates split candidates at midpoints between sorted unique feature values.
- Chooses the split with the lowest weighted child-node Gini impurity.
- Recursively builds a binary tree until the configured maximum depth, a pure node, or insufficient samples.
- Implements recursive class prediction and leaf-frequency-based probability prediction.
- Supports randomly sampled split thresholds for use by Extra Trees.

### Custom Random Forest — `randomForest.py`

- Creates **bootstrap samples** of training examples, with replacement.
- Selects a random subset of input features for each tree.
- Fits custom Decision Trees and aggregates predictions through **majority voting**.
- Averages individual tree class-probability estimates.

### Custom Extra Trees — `extraTree.py`

- Uses the **entire training dataset** per tree (no bootstrap sampling in this implementation).
- Randomly selects a subset of features for each tree.
- Uses **random split thresholds**, then selects the lowest-impurity split among the generated candidates.
- Aggregates predictions using majority voting and averages class probabilities.

### Utility functions — `helper.py`

Contains helpers for bootstrap row sampling and random feature-subset selection.

### Experimental setup and comparison

The `main.py` experiment runs on the built-in **Iris** and **Wine** datasets from scikit-learn.

| Setting | Value |
| --- | --- |
| Train/test split | 70% / 30% |
| Custom Decision Tree max depth | Iris: 2; Wine: 6 |
| Number of trees in custom ensembles | 5 |
| Random Forest bootstrap helper argument | `min_samples_split=10` (used for determining bootstrap sample size, not passed as a tree stopping rule) |
| scikit-learn comparison | `DecisionTreeClassifier`, `RandomForestClassifier(n_estimators=100)`, `ExtraTreesClassifier(n_estimators=100)` |
| Metrics | **Accuracy, macro F1-score, multiclass one-vs-rest AUROC** |

Run the script to print the performance of all **six model variants** (three custom implementations and three scikit-learn models) on **both datasets**. Numerical scores are intentionally not hard-coded here; reproduce them by running the experiment.

## Technologies used

- **Language:** Python
- **Data processing:** NumPy, pandas
- **Visualization:** Matplotlib, Seaborn
- **Machine learning:** scikit-learn (preprocessing, splitting, baselines, metrics, built-in datasets)
- **Deep learning:** PyTorch (`torch`, `torch.nn`, `torch.utils.data`)
- **Notebooks:** Jupyter Notebook

## Getting started

Clone the repository:

```bash
git clone https://github.com/nabila-mumu/ml_coursework.git
cd ml_coursework
```

Install the Python dependencies (preferably in a virtual environment):

```bash
python -m pip install numpy pandas matplotlib seaborn scikit-learn torch jupyter
```

**Run the tree-based experiments** from the repository root:

```bash
cd decision_tree_offline
python main.py
```

**Run the preprocessing notebooks** (in a separate terminal, from the repository root):

```bash
cd data_preprocessing_offline
jupyter notebook
```

Open `iris_classification.ipynb`, `Feature_non_numeric.ipynb`, or `2005050/2005050.ipynb`. For the nested `2005050.ipynb`, ensure the notebook can find `medical_students_dataset.csv` in its working directory; that CSV is stored one directory above the notebook. For example, copy the CSV into `data_preprocessing_offline/2005050/` before running that notebook, or adjust its `pd.read_csv(...)` path.

## Assignment documents and reports

- [Data preprocessing assignment specification](data_preprocessing_offline/July25_CSE472_Assignment1.pdf)
- [Data preprocessing report](data_preprocessing_offline/2005050/2005050_report.pdf)
- [Decision tree assignment specification](decision_tree_offline/CSE_472_Assignment_2_DT.pdf)
- [Decision tree report](decision_tree_offline/2005050_report.pdf)

## Skills demonstrated

- Missing-value handling, duplicate removal, categorical encoding, and scaling
- Exploratory visualization, correlation analysis, and feature selection
- Feed-forward network training, validation, and classification metrics
- Implementing impurity-based tree induction and ensemble learning **without relying on scikit-learn model implementations**
- Comparing custom classifiers with established library baselines

---
