# Soft Computing Project: Learning Fuzzy Logic Gates with Neural Nets

Trains small neural networks to approximate five fuzzy-logic operators (AND, OR,
NOT, XOR, and Implication) from sampled `(X1, X2) -> Target` data in the included
`.xlsx` files. Part 1 approaches this as regression (`MLPRegressor` + `GridSearchCV`);
Part 2 approaches it as classification (`MLPClassifier`), which fits better since
`Target` is actually binary (0/1) in every dataset.

## Setup

```
pip install -r requirements.txt
```

Open either notebook in Jupyter, JupyterLab, or Colab. Both notebooks look for the
`.xlsx` files in the same folder they're run from (`DATA_DIR = "."`), so no need to
unzip anything unless you're working from an `archive.zip`, in which case just drop
it next to the notebook and the first cell will extract it for you.

## Files

- `PE-SC Project Part 1.ipynb`: regression approach
- `PE-SC Project Part 2.ipynb`: classification approach
- `*.xlsx`: the five fuzzy-operator datasets (1000 rows each)

## What was fixed vs. the original notebooks

1. **AND/OR classifiers were swapped on save/load (Part 2).** The original pickled
   `model_and_op` under a filename starting with `OR`, and `model_or_op` under a
   filename starting with `and_op`. After reloading, `model_and_op` held the OR
   network and vice versa, so the final "Fuzzy AND Result" / "Fuzzy OR Result"
   printout was showing each other's predictions. Every model is now saved and
   loaded under a name that matches what it actually is (`AND_Classifier.sav`,
   `OR_Classifier.sav`, etc.).
2. **The Implication regression model only saw `X1` (Part 1).** Implication is a
   function of two inputs, so a model trained on one input can't learn it. It now
   trains on both `X1` and `X2`.
3. **Evaluation metrics were computed on the wrong array (Part 1).** The original
   inverse-transformed the *scaled X test values* instead of the model's actual
   predictions before scoring, so the reported RMSE/MSE/MAE/R² didn't measure
   model accuracy at all. Fixed to inverse-transform `y_predict`.
4. **One `MinMaxScaler` was reused for both X and y**, and refitting it on the
   second variable silently overwrote what it had learned from the first. Now uses
   separate `x_scaler` / `y_scaler` instances throughout.
5. **Added classification metrics (accuracy, F1, confusion matrix) to Part 1**
   alongside RMSE/R², since `Target` is binary; the regression metrics alone
   don't have a clean interpretation on a two-class target.
6. **Removed a stray leading-space indentation** on the OR-model cell in Part 2.
   IPython/Colab silently tolerates a uniformly-indented cell, but the same code
   raises `IndentationError` in a plain script.
7. **De-duplicated five near-identical cells** in Part 1 (each loaded one file and
   found the two rows with the smallest `min(X1, X2)`) into one `smallest_two()`
   function.
8. **Removed Colab-only hardcoded paths** (`/content/...`) and a redundant
   duplicate save of the Implication model under a second, typo'd filename.
9. **Part 1's Implication and XOR sections shared variable names** (`scaler`,
   `mlp_grid`), since the XOR section silently overwrote the Implication model by the
   time the notebook finished. Renamed to `impl_*` / `xor_*` so both survive.

## New features added after the initial fix pass

- **Manual input.** Both notebooks now have a "try your own values" cell: edit
  `x1_val` / `x2_val` (0 to 1) and re-run to get that model's prediction. Part 1
  has one for the Implication model and one for the XOR model; Part 2 has one that
  runs all 5 reloaded classifiers (and, further down, the shared network) on the
  same pair.
- **Shared multi-output network (Part 2).** One `MLPClassifier` predicts all 5
  gates at once from `(X1, X2)`, trained on the AND/OR/XOR/Implication data pooled
  together (`sklearn` handles a multi-column 0/1 target natively). Compared
  against the 5 separate networks on a held-out split of that pooled data.
- **Classical (no-learning) baseline**, in both notebooks. Since every dataset's
  `Target` was confirmed to be an exact threshold-at-0.5 Boolean rule (verified
  directly against the data), computing that rule directly is definitionally
  perfect. The comparison isn't "does it beat the rule," it's "how close does the
  learned network get." Part 2's version checks this on 300 fresh random points
  none of the models were trained on, which also let the individual vs. shared
  networks be compared on true held-out data: in one run, the shared network (more
  training data, since it pools 4 files) generalized noticeably better than the
  individual per-gate networks (99-100% vs. 91-97%), a real result, not a
  guaranteed one; your numbers will vary run to run.
- **Data-size control (Part 2).** Retrains just the NOT model on 3,200 rows
  (matching the shared network's training set size, same architecture) to check
  whether the shared network's edge above is really about learning several gates
  together, or just about seeing more data. In testing, the data-matched NOT model
  reached the same or better accuracy as the shared network on its own, so here,
  it's mostly a data-size effect, not evidence of a multi-task learning benefit.
- **Noise robustness check (Part 2).** Perturbs the same 300 check points with
  Gaussian noise at several levels, feeds the noisy values to the classical
  formula and to the trained networks, and scores both against the original clean
  labels. In testing, the shared network's accuracy tracked the classical
  formula's almost exactly at every noise level; it did not degrade more
  gracefully. That makes sense on reflection: every training label was a clean,
  unambiguous 0/1 with nothing near the boundary marked uncertain, so the network
  had no reason to learn anything softer than the same hard cutoff.
- **Training on noisy inputs (Part 2).** Went a step further and actually trained
  a second shared network on noise-augmented inputs (clean copy plus copies at
  sigma = 0.03/0.06/0.1, all labelled with the original clean value's true label),
  to see if that unlocks a real edge. It didn't, reliably. There's a real
  mathematical reason: when the true label is a hard step function of the input
  and the measurement noise is symmetric, thresholding the noisy reading at 0.5
  is already the error-minimizing decision; no classifier, learned or not, can
  systematically beat it under this exact setup. A genuine advantage would need
  either a true relationship that's softer than a hard step, or noise that's
  asymmetric/input-dependent in a way a fixed threshold can't adapt to.
- **Noise sweep extended to 80% (Part 2).** All approaches (classical, individual,
  shared, noise-trained) stay tied at every level from 0% to 80%; past ~60% noise
  they all converge to roughly the ~66% majority-class floor, since the input has
  become nearly uninformative by then.
- **Rescaling/smoothing the noisy input (Part 2).** Tested whether re-stretching
  the noisy batch with min-max normalization, or replacing the hard `[0, 1]` clip
  with a smooth sigmoid, recovers any accuracy. Neither did, at any noise level: both are
  monotonic transforms of a single already-noisy reading, and a
  monotonic transform can't add back information that noise already destroyed.
- **Averaging repeated noisy readings (Part 2).** This one actually works: taking
  `k` independent noisy draws of the *same* true `(X1, X2)` and averaging them
  before classifying steadily recovers accuracy as `k` grows, at every noise level
  tested (including 80%), because it's adding genuine new information (more
  independent looks at the truth), not just reshaping one noisy number. The
  catch: it only applies if a real system can actually take `k` repeated
  measurements of the same thing before deciding, not `k` different situations.
- **"What this shows" notes.** Every result-producing cell in both notebooks now
  has a short markdown note right after it stating the actual takeaway (with real
  numbers from testing where those numbers are stable, and qualitative/range
  language where a result is expected to vary a bit run to run), so the notebooks
  read as a self-contained writeup rather than code with only numbers attached.
