# Soft Computing Project: Learning Fuzzy Logic Gates with Neural Nets
 
Trains small neural networks to approximate five fuzzy-logic operators (AND, OR,
NOT, XOR, and Implication) from sampled `(X1, X2) -> Target` data in the included
`.xlsx` files. Part 1 approaches this as regression (`MLPRegressor` + `GridSearchCV`);
Part 2 approaches it as classification (`MLPClassifier`), which fits better since
`Target` is actually binary (0/1) in every dataset.
 
## Setup
  
Open either notebook in Jupyter, JupyterLab, or Colab. Both notebooks look for the
`.xlsx` files in the same folder they're run from (`DATA_DIR = "."`), so no need to
unzip anything unless you're working from an `archive.zip`, in which case just drop
it next to the notebook and the first cell will extract it for you.
 
## Files
 
- `PE-SC Project Part 1.ipynb`: regression approach
- `PE-SC Project Part 2.ipynb`: classification approach
- `*.xlsx`: the five fuzzy-operator datasets (1000 rows each)
