# Prompt Injection Guard — Notebook Explained

This document walks through `01_EDA.ipynb` one section at a time: what
each line does mechanically, the underlying concept, and why it matters
for this project specifically. Read this alongside the notebook — each
heading here matches a heading in the notebook.

---

# Step 1: EDA

---

## Imports

```python
import pandas as pd
```
**Pandas** is the standard library for tabular data in Python. A CSV file
becomes a `DataFrame` — a spreadsheet you manipulate with code instead of
a mouse. Aliased `pd` by near-universal convention; every ML codebase you
read will do this, so match it.

```python
import numpy as np
```
**NumPy** is the numerical engine pandas is built on top of (arrays,
vectorized math). This particular script never calls `np.` directly —
pandas uses it under the hood. Flagging this as an unused import: a habit
worth building is removing imports you don't actually call, since a
linter (like `flake8`) will warn about it and it's noise for a reader.

```python
import matplotlib.pyplot as plt
```
**Matplotlib** is the plotting library. `pyplot` is a stateful interface:
you build a chart step by step (create a figure → add axes → plot data →
save). Aliased `plt` by convention.

```python
from collections import Counter
```
`Counter` is a dict subclass built for counting things. Feed it a list of
words, it tallies occurrences, and `.most_common(n)` gives you the top n
without writing a sort yourself. Used below to find frequent words per
class.

```python
import re
```
Python's **regular expressions** module. Used to extract word-like tokens
from raw text — regex lets you describe a *pattern* ("a run of 3+
lowercase letters") rather than writing manual character-by-character
parsing.

---

## Loading the data

```python
df = pd.read_csv("data/hf_prompt_injections.csv")
```
`read_csv` parses the file into a DataFrame. `df` is the near-universal
variable name for "the dataframe I'm currently working with" — you'll
see it in almost every pandas script you ever read.

---

## Shape and missingness check

```python
print(f"Rows: {df.shape[0]}, Cols: {df.shape[1]}")
```
`.shape` returns a tuple `(n_rows, n_cols)`. This is always the first
thing to check on a new dataset — it tells you the scale you're working
with before anything else.

```python
print(df.isna().sum())
```
`.isna()` returns a same-shaped DataFrame of `True`/`False` — `True`
wherever a cell is missing (NaN). `.sum()` on a boolean column counts
the `True`s (Python treats `True` as `1`), so this line gives you a
missing-value count **per column** in one line. This matters because
missing values crash most ML functions if you don't handle them —
checking this first tells you whether you need an imputation step at
all. Here, the answer was zero across all columns, which is unusually
clean for real-world text data.

```python
print(f"\nDuplicate rows: {df.duplicated().sum()}")
```
`.duplicated()` returns `True` for any row that is an exact repeat of an
earlier row (checking *all* columns at once). Duplicates matter because
if the same example appears in both your training and test split later,
your model gets an unfair "preview" of the test set — this is called
**data leakage**, and it makes your evaluation metrics lie to you.

```python
print(f"Duplicate text: {df['text'].duplicated().sum()}")
```
Same idea, but narrowed to just the `text` column via `df['text']`
(selecting a single column from a DataFrame gives you a `Series` — a
1-dimensional version of a DataFrame). This check is stricter: two rows
could have identical text but a different label by data-entry error, and
this would still catch it as a duplicate text even if the full-row
duplicate check above didn't.

---

## Class balance

```python
print(df['label'].value_counts())
```
`.value_counts()` counts how many times each unique value appears in a
column, sorted descending. For a binary label column, this immediately
tells you if your classes are balanced (50/50) or skewed.

```python
print(df['label'].value_counts(normalize=True).round(3))
```
`normalize=True` converts the counts to proportions (summing to 1.0)
instead of raw counts. `.round(3)` just trims the decimal for
readability. **Why this matters more than it seems**: if one class is
90% of your data, a model that *always* predicts that class scores 90%
accuracy while doing zero useful work. Knowing the baseline split before
training tells you what "better than doing nothing" actually looks like
for this dataset (here: 57/43, so a naive model would hit ~57% accuracy
by always guessing "benign").

---

## Source dataset breakdown

```python
print(pd.crosstab(df['source_dataset'], df['label']))
```
`pd.crosstab` builds a contingency table — rows are one column's unique
values, columns are another's, cells are counts of how often that
combination occurs together. Here it answers: "does each source dataset
contribute both classes, or does one source only supply benign examples
and the other only injections?" If it were the latter, your model could
cheat by learning to recognize the *source dataset's writing style*
rather than actually learning what an injection looks like — another
form of leakage, this time via a hidden shortcut rather than a literal
duplicate.

---

## Engineering length features

```python
df['char_len'] = df['text'].astype(str).str.len()
```
This creates a **new column** by assigning to `df['char_len']` — pandas
lets you add columns just by assigning to a name that doesn't exist yet.
`.astype(str)` forces every value to a string first (defensive — avoids
a crash if any cell were accidentally a non-string type, like `NaN`
which is technically a float). `.str.len()` is a *vectorized string
method*: it applies Python's `len()` to every row's text at once,
without writing a manual loop. Result: character count per row.

```python
df['word_len'] = df['text'].astype(str).str.split().str.len()
```
Same pattern, but `.str.split()` first breaks each string into a list of
words on whitespace, then `.str.len()` counts the *length of that list*
— i.e., word count, not character count. This is a genuinely new
feature we're engineering by hand, not something in the original data —
worth noticing, because feature engineering (creating new signal from
raw columns) is a core ML skill, separate from just calling
`.fit()` on whatever columns already exist.

**Why length at all**: length is cheap to compute and, per the results,
turned out to correlate with the label (injections run longer on
average) — so it's a candidate input feature for the model later, not
just a descriptive stat.

---

## Grouped statistics

```python
print(df.groupby('label')[['char_len', 'word_len']].describe().T)
```
`.groupby('label')` splits the DataFrame into two groups (one per label
value) without physically copying anything yet — it's a lazy operation.
Selecting `[['char_len', 'word_len']]` narrows to just those two columns
within each group. `.describe()` then computes count, mean, std, min,
quartiles, and max **per group**. `.T` transposes the result (swaps rows
and columns) purely so it prints more readably in a terminal — no
statistical meaning, just formatting. This is how we found that
injection-class text averages more words than benign text.

---

## Building the 2×2 plot grid

```python
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
```
`plt.subplots(2, 2)` creates one figure containing a 2-row-by-2-column
grid of individual chart areas ("axes" in matplotlib terminology — each
subplot is an `Axes` object). It returns two things: `fig` (the overall
canvas/container) and `axes` (a 2×2 NumPy array of the four individual
plot areas, indexed like `axes[0, 0]` for top-left). `figsize=(13, 10)`
sets the overall canvas size in inches. We build four related plots in
one image rather than four separate images so they can be compared
side-by-side in a single glance.

```python
df['label'].value_counts().plot(kind='bar', ax=axes[0, 0], color=['#4C72B0', '#C44E52'])
```
Pandas Series objects have a built-in `.plot()` method — a shortcut over
calling matplotlib directly. `kind='bar'` requests a bar chart.
`ax=axes[0, 0]` tells it to draw into the top-left subplot specifically,
rather than creating a new figure (without this, pandas would ignore our
2×2 grid entirely). The `color` list assigns blue to the first bar and
red to the second, by position.

```python
axes[0, 0].set_title('Class Balance (0=benign, 1=injection)')
axes[0, 0].set_xticklabels(['Benign (0)', 'Injection (1)'], rotation=0)
```
Direct method calls on the `Axes` object to add a title and relabel the
x-axis tick marks with human-readable text instead of raw `0`/`1`.
`rotation=0` keeps the labels horizontal (pandas sometimes auto-rotates
them, which looks worse for only two short labels).

```python
for label, color in [(0, '#4C72B0'), (1, '#C44E52')]:
    subset = df[df['label'] == label]['word_len']
    axes[0, 1].hist(subset, bins=40, alpha=0.6, label=f'label={label}', color=color, range=(0, 150))
```
This loop draws two overlapping histograms on the same axes so their
shapes can be compared directly. `df[df['label'] == label]` is
**boolean indexing**: `df['label'] == label` produces a `True`/`False`
Series the same length as the DataFrame, and wrapping the DataFrame in
those brackets keeps only the rows where it's `True` — this is the
single most common pattern for filtering in pandas, worth memorizing.
`['word_len']` then grabs just that column from the filtered rows.
`bins=40` splits the value range into 40 buckets; `alpha=0.6` makes the
bars semi-transparent so overlapping regions of the two histograms are
visible instead of one fully hiding the other; `range=(0, 150)` clips
the x-axis so a few extreme outliers (one sample was 783 words) don't
squash the entire useful part of the chart into a sliver.

```python
axes[0, 1].set_title('Word Count Distribution by Class')
axes[0, 1].set_xlabel('Words per sample')
axes[0, 1].legend()
```
Title and axis label as before. `.legend()` draws the box identifying
which color is which class, using the `label=` values we set in the
loop above.

```python
pd.crosstab(df['source_dataset'], df['label']).plot(kind='bar', stacked=True, ax=axes[1, 0], color=['#4C72B0', '#C44E52'])
```
Reuses the same crosstab from earlier, but this time plots it directly
as a **stacked bar chart** (`stacked=True` means the two label-counts
for each source dataset are drawn on top of each other in one bar,
rather than side by side) — a compact way to show both the total volume
per source and its class composition in one glance.

```python
axes[1, 0].tick_params(axis='x', rotation=20)
```
Rotates the x-axis tick labels 20 degrees since the dataset names are
long strings that would otherwise overlap each other.

```python
df.boxplot(column='char_len', by='label', ax=axes[1, 1])
axes[1, 1].set_yscale('log')
axes[1, 1].set_title('Char Length by Class (log scale)')
plt.suptitle('')
```
`.boxplot()` draws a box-and-whisker plot: the box shows the
25th–75th percentile range, the line inside is the median, and
whiskers/dots show spread and outliers. `by='label'` automatically
splits it into one box per class. `set_yscale('log')` switches the
y-axis to a logarithmic scale — necessary here because a handful of
extreme outliers (up to ~4,500 characters) would otherwise compress
the entire "normal" range of values into an unreadable sliver at the
bottom. `plt.suptitle('')` clears an automatic overall title that
`.boxplot()` adds by default (it would otherwise duplicate/clash with
our own `set_title`).

```python
plt.tight_layout()
plt.savefig('notebooks/eda_overview.png', dpi=110)
```
`tight_layout()` automatically adjusts spacing between the four subplots
so titles/labels don't overlap each other. `savefig` writes the whole
figure to a PNG file; `dpi=110` sets resolution (higher = crisper but
larger file).

---

## Word frequency analysis

```python
def top_words(texts, n=20):
    words = []
    for t in texts:
        words.extend(re.findall(r'[a-z]{3,}', str(t).lower()))
    return Counter(words).most_common(n)
```
A function definition — `texts` is expected to be an iterable of strings
(we'll pass a pandas Series, which is iterable). `words = []` starts an
empty list to accumulate every word found across all texts. Inside the
loop: `str(t).lower()` forces lowercase so "Ignore" and "ignore" count
as the same word. `re.findall(r'[a-z]{3,}', ...)` is a regex pattern
meaning "find every run of 3 or more lowercase letters" — this
extracts word-like tokens while automatically discarding punctuation,
numbers, and 1–2 letter noise (like "a", "an", "is") without needing a
separate cleaning step. `.extend()` (not `.append()`) adds each found
word individually to the running list, rather than adding one sublist
per text. Finally `Counter(words).most_common(n)` tallies and returns
the top n. **Note**: this function is defined but actually superseded
by `top_words_clean` below — it's a leftover exploratory step. Worth
knowing: real analysis code often keeps early drafts around briefly;
cleaning them out is part of turning a script into something
presentable.

```python
STOP = set("""the a an and or of to in for is are was were with this that on as be
your you it its i we our not have has had do does can will would should
""".split())
```
A **stopword list** — extremely common words that carry little topical
meaning ("the", "is", "and") and would otherwise dominate any word
frequency count regardless of what the text is actually about.
`"""..."""` is a multi-line string; `.split()` breaks it on whitespace
into individual words; wrapping in `set(...)` converts the list to a
set, which makes membership checks (`word not in STOP`, used next) much
faster than checking against a list, especially as the stopword list
grows. This is a small hand-written list rather than a library import
(like NLTK's stopword list) — a reasonable shortcut for a quick EDA
pass, but worth swapping for a proper library list if this became
production code, since hand-written lists miss edge cases.

```python
def top_words_clean(texts, n=20):
    words = []
    for t in texts:
        words.extend(w for w in re.findall(r'[a-z]{3,}', str(t).lower()) if w not in STOP)
    return Counter(words).most_common(n)
```
Same as `top_words`, but the `for w in ... if w not in STOP` is a
**generator expression** filtering out stopwords inline before they're
added to `words`. This is the function actually used for the printed
results — it's what let "what", "how", "explain" surface for benign
text and "reveal", "bypass", "ignore" surface for injections, instead
of both lists just being full of "the", "is", "to".

```python
print(top_words_clean(df[df['label'] == 0]['text']))
print(top_words_clean(df[df['label'] == 1]['text']))
```
Same boolean-indexing pattern as before, filtering to each class, then
passing that filtered text column into the function. This is the step
that produced the single most useful finding in the whole EDA: the
vocabulary that separates the two classes is dominated by
constraint-circumvention words ("bypass", "ignore", "restrictions") on
the injection side — direct evidence that a bag-of-words model should
work reasonably well as a first pass, before reaching for anything
heavier like a transformer.

---

## Eyeballing raw examples

```python
for t in df[df['label'] == 1]['text'].sample(5, random_state=42):
    print("-", str(t)[:150])
```
`.sample(5, ...)` pulls 5 random rows from the filtered Series.
`random_state=42` seeds the random number generator — without it, you'd
get a different random sample every time you re-run the script, making
it impossible to discuss "the same" examples with someone else or
reproduce a result later. `42` has no special meaning beyond being a
common convention (a nod to *Hitchhiker's Guide to the Galaxy*) — any
fixed integer works identically. `[:150]` truncates each string to its
first 150 characters so long samples don't flood the terminal.

**Why this step matters even after all the statistics above**: numbers
and charts tell you *that* a pattern exists, but reading actual examples
is what tells you *what the pattern looks like in practice* — and
sometimes reveals things aggregate stats hide entirely (e.g., you might
notice injection examples that don't use any "attack" vocabulary at
all, which would warn you the model will likely miss those).

---

## Summary of what this script established

1. Data is clean — no missing values, negligible duplicates.
2. Classes are mildly imbalanced (57/43) — accuracy alone won't be a
   trustworthy metric later; we'll need precision/recall/F1.
3. Both source datasets contribute both classes — no leakage risk via
   dataset-of-origin.
4. Injection text runs longer on average — a usable engineered feature.
5. Vocabulary clearly differs by class, centered on
   constraint-circumvention language — strong signal that a simple
   TF-IDF + linear model baseline is a reasonable starting point,
   which is exactly what Step 2 builds.

# Step 2: Baseline Model

---

## Additional imports for modeling

```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
import joblib
```
`train_test_split` splits data into training and test sets. `TfidfVectorizer`
converts raw text into numeric features (explained in detail below).
`LogisticRegression` and `MultinomialNB` are the two baseline model types
we'll compare. The `metrics` imports are all evaluation tools. `joblib` is
a library for saving trained Python objects (models, vectorizers) to disk
so you can load them later without retraining — this is what a real
backend service will do in Step 5.

---

## Train/test split

```python
X = df['text']
y = df['label']
```
Convention: `X` (capital, since it's typically 2D/tabular) holds the
input features, `y` (lowercase, typically 1D) holds the target you're
trying to predict. Here `X` is just the raw text column — it isn't
numeric yet, so it isn't usable by a model yet. That conversion happens
next, in the TF-IDF step, deliberately kept separate from the split.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```
`train_test_split` shuffles and divides your data. `test_size=0.2`
means 20% of rows go to the test set, 80% to training — a standard
default; enough test data to trust the evaluation, without starving the
model of training examples. `random_state=42` seeds the shuffle for
reproducibility (same reason as the `.sample()` calls in Step 1).

**`stratify=y` is the important one to understand**: without it, a
random split could — by chance — put unusually few injection examples
in your test set, making your evaluation metrics noisy or misleading.
`stratify=y` forces the split to preserve the *same class ratio*
(57/43) in both the train and test sets. This is standard practice
whenever you have any class imbalance, however mild.

The function returns four objects at once, in this fixed order:
train features, test features, train labels, test labels — worth
memorizing the order, since getting it wrong silently produces garbage
results without necessarily erroring.

```python
print("Train size:", len(X_train), "Test size:", len(X_test))
print(y_train.value_counts(normalize=True).round(3))
print(y_test.value_counts(normalize=True).round(3))
```
Sanity checks — confirming the split sizes and, critically, confirming
`stratify` actually worked (both should show ~57/43). **Always verify
your assumptions like this rather than trusting a parameter silently
did what you expect.**

---

## TF-IDF vectorization

This is the step that turns text into numbers a model can actually
learn from — arguably the most important concept in this whole
baseline.

```python
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
```
**TF-IDF** stands for Term Frequency–Inverse Document Frequency. The
core idea: represent each text sample as a vector of numbers, one
number per word (or word-pair) in your vocabulary, where the number
reflects how *important* that word is to that specific sample — not
just how often it appears.

- **Term Frequency** — how often a word appears in *this* sample.
- **Inverse Document Frequency** — a downweighting factor for words
  that appear in *many* samples across the whole dataset (like "the"
  or "is") since a word appearing everywhere carries little information
  about what makes any one sample distinctive. A word that's rare
  overall but concentrated in one sample gets a high score.

Parameters:
- `max_features=5000` caps the vocabulary to the 5,000 highest-value
  terms, dropping the long tail of rare/noisy words — keeps the
  resulting matrix a manageable size and reduces overfitting risk from
  ultra-rare terms that appear in only one or two samples.
- `ngram_range=(1, 2)` means "use both single words (unigrams) AND
  two-word phrases (bigrams)" as vocabulary entries. This is why
  `"prompt injection"` and `"ai safety"` showed up as their own
  features later — a bigram captures phrase-level meaning that
  individual words miss (e.g. "ai" and "safety" separately don't imply
  the same thing as "ai safety" together).
- `stop_words='english'` removes a built-in list of common English
  filler words automatically — the library equivalent of the hand-rolled
  `STOP` set from Step 1's EDA, just more complete.

```python
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
```
This is one of the most important patterns in all of applied ML, worth
sitting with: **`fit_transform` on train, `transform` (only) on test.**

`fit_transform` does two things at once on the training data: it
*learns* the vocabulary and the IDF weights from `X_train` (the "fit"
part), and then converts `X_train` into numeric vectors using what it
just learned (the "transform" part).

`transform` (without `fit`) on the test data reuses the *exact same*
vocabulary and weights learned from training — it does **not** relearn
anything from the test set. This is deliberate and critical: if you let
the vectorizer see the test data during fitting, information about the
test set would leak into your feature representation, and your
evaluation would no longer honestly reflect how the model performs on
truly unseen data. This is the single most common data leakage bug
beginners introduce, and it's worth internalizing this pattern for
every preprocessing step you ever write (scalers, encoders, etc. all
follow this same fit-on-train-only rule).

```python
print("TF-IDF matrix shape (train):", X_train_tfidf.shape)
```
Shape will be `(9278, 5000)` — 9,278 training samples (rows), 5,000
vocabulary features (columns). Each row is now a numeric vector
representing one text sample, ready to feed into a model.

---

## Model 1 — Logistic Regression

```python
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train_tfidf, y_train)
```
**Logistic Regression** is a linear model for classification — despite
the name, it's used for classification, not regression. Conceptually,
it learns one weight per feature (here, per TF-IDF vocabulary term),
and combines them into a probability that a given input belongs to the
positive class. It's an excellent first model for text classification
because TF-IDF vectors are high-dimensional and often close to linearly
separable — meaning a straight-line-style decision boundary already
captures most of the signal, without needing a more complex model.

`max_iter=1000` raises the cap on optimization iterations from
sklearn's default (100) — text data with thousands of features
sometimes needs more iterations to fully converge; without this you'd
likely see a convergence warning. `random_state=42` seeds the
optimizer's internal randomness for reproducibility, same reasoning as
before. `.fit()` is the actual training step — the model looks at every
training example and its label, and adjusts its internal weights to
minimize prediction error.

```python
logreg_preds = logreg.predict(X_test_tfidf)
logreg_probs = logreg.predict_proba(X_test_tfidf)[:, 1]
```
`.predict()` returns hard class labels (0 or 1) for the test set —
whatever the model's most likely answer is. `.predict_proba()` instead
returns *probabilities* for both classes as a 2-column array; `[:, 1]`
selects just the second column (probability of class 1, "injection").
We need both: hard predictions for the classification report, and raw
probabilities for the ROC curve (explained below), which needs a
continuous score rather than a hard yes/no to work.

```python
print(classification_report(y_test, logreg_preds, target_names=['benign', 'injection']))
```
`classification_report` prints precision, recall, F1-score, and support
(sample count) for each class, plus overall accuracy. This is the
standard, most important evaluation output for any classifier —
**precision** answers "of everything I flagged as injection, how much
actually was?", **recall** answers "of everything that actually was an
injection, how much did I catch?", and **F1** is their harmonic mean
(a single balanced summary of both). In a security context, recall
on the injection class deserves the most attention — a missed attack
(false negative) is typically more costly than a false alarm.

```python
print("ROC-AUC:", round(roc_auc_score(y_test, logreg_probs), 4))
```
**ROC-AUC** (Area Under the Receiver Operating Characteristic curve)
summarizes a model's ability to *rank* positive examples above negative
ones across every possible decision threshold, into a single number
between 0.5 (no better than random guessing) and 1.0 (perfect ranking).
It's threshold-independent — unlike precision/recall, which depend on
where you draw the "yes/no" cutoff — which makes it a good single
number for comparing two models overall, before you've decided exactly
how conservative or aggressive you want the final system to be.

---

## Model 2 — Naive Bayes

```python
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
```
**Multinomial Naive Bayes** is a probabilistic classifier built on
Bayes' theorem, with a deliberately simplifying ("naive") assumption:
that every feature (word) contributes to the prediction independently
of every other feature. That assumption is technically false for
language (word order and co-occurrence obviously matter), but the
model is fast, needs very little data to train well, and is a classic,
strong baseline for text classification specifically — worth having in
your toolkit as a first thing to try, precisely because it's cheap and
often surprisingly competitive.

The rest of the block mirrors the Logistic Regression evaluation
exactly (`.predict()`, `.predict_proba()`, `classification_report`,
`roc_auc_score`) — reusing the identical evaluation code for both
models is intentional: it's the only way to make the comparison
between them fair and legible.

**Result worth noting**: Naive Bayes edged out Logistic Regression on
injection-class recall (90% vs 87%) despite near-identical accuracy and
ROC-AUC. This is a good early lesson that the "better" model depends on
which metric you actually care about for the use case — not just which
one has the higher headline accuracy number.

---

## Evaluation plots — confusion matrices + ROC comparison

```python
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
```
Same pattern as Step 1's plotting: one figure, a row of 3 subplots this
time instead of a 2×2 grid. **Important lesson carried over from Step
1's bug**: all three panels are built and shown within this single
cell, start to finish — splitting figure construction across multiple
cells is what broke the earlier notebook.

```python
ConfusionMatrixDisplay.from_estimator(logreg, X_test_tfidf, y_test, ax=axes[0], cmap='Blues',
                                        display_labels=['benign', 'injection'])
axes[0].set_title('Logistic Regression — Confusion Matrix')
```
A **confusion matrix** is a 2×2 grid (for binary classification)
showing counts of: true negatives (top-left, correctly predicted
benign), false positives (top-right, predicted injection but was
actually benign), false negatives (bottom-left, predicted benign but
was actually injection), and true positives (bottom-right, correctly
predicted injection). `ConfusionMatrixDisplay.from_estimator` is a
convenience method that runs prediction *and* builds the plot in one
call, rather than you manually calling `confusion_matrix()` and then
plotting the array yourself. `cmap='Blues'` sets the color scale.

The same call repeats for `nb` with a different color (`Oranges`) purely
so the two matrices are visually distinguishable side by side.

```python
RocCurveDisplay.from_estimator(logreg, X_test_tfidf, y_test, ax=axes[2], name='Logistic Regression')
RocCurveDisplay.from_estimator(nb, X_test_tfidf, y_test, ax=axes[2], name='Naive Bayes')
```
Both calls target the *same* `axes[2]`, which is what overlays both
curves on one chart for direct comparison, rather than each producing
its own separate plot. A **ROC curve** plots the true positive rate
against the false positive rate as the decision threshold sweeps from
0 to 1 — a curve that hugs the top-left corner indicates a strong
model; the diagonal line represents random guessing.

```python
axes[2].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random guess')
axes[2].legend()
```
Manually draws that diagonal reference line (a straight line from
(0,0) to (1,1)) so the "random guessing" baseline is visible on the
chart, not just implied.

---

## Inspecting top predictive features

```python
feature_names = vectorizer.get_feature_names_out()
coefs = logreg.coef_[0]
```
`get_feature_names_out()` returns the actual vocabulary words/phrases
the vectorizer learned, in the same order as the columns of the TF-IDF
matrix — this lets us map numeric feature indices back to human-readable
words. `logreg.coef_` holds the learned weight for every feature; `[0]`
takes the first (and only, since this is binary classification) row of
weights. A **positive** coefficient means that word's presence pushes
the prediction toward class 1 (injection); a **negative** coefficient
pushes toward class 0 (benign).

```python
top_injection_idx = coefs.argsort()[-15:][::-1]
top_benign_idx = coefs.argsort()[:15]
```
`.argsort()` returns the *indices* that would sort the array ascending
(smallest coefficient first). `[-15:]` takes the last 15 indices (the
15 largest, most positive coefficients — i.e. strongest injection
signals), and `[::-1]` reverses that slice so it prints
largest-first instead of smallest-first. `[:15]` takes the first 15
indices directly (the most negative coefficients — strongest benign
signals) with no reversal needed since ascending order is already
"most negative first."

**What this revealed**: words like "restrictions", "guidelines",
"unrestricted", and "reveal" carry the strongest injection signal —
consistent with the raw word-frequency finding from Step 1's EDA, but
now quantified with an actual learned weight rather than just a raw
count. More interesting: "prompt injection" and "ai safety" (as
bigrams) push toward **benign** — this makes sense once you remember
some benign samples are literally meta-questions *about* prompt
injection as a topic (e.g. "what is prompt injection and how does it
work") rather than actual attack attempts. This is exactly the
vocabulary overlap flagged as a risk back in Step 1 — now visible
directly in the model's learned weights, not just a theoretical
concern.

---

## Saving the model artifacts

```python
joblib.dump(logreg, '../models/logreg_baseline.joblib')
joblib.dump(vectorizer, '../models/tfidf_vectorizer.joblib')
```
`joblib.dump` serializes a Python object to a file on disk — here, both
the trained model *and* the fitted vectorizer. **Both need saving, not
just the model**: at inference time (when the API receives a new prompt
to check), you'll need to run that new text through the *exact same*
fitted vectorizer before the model can use it — a freshly-created
vectorizer wouldn't have the same vocabulary/weights learned from
training, and predictions would be meaningless. This pair of files is
exactly what Step 5's FastAPI service will load at startup.

---

## Step 2 summary

1. Stratified 80/20 split keeps the class ratio consistent across
   train and test.
2. TF-IDF converts text into numeric vectors, fit only on training
   data to avoid leakage — the single most important pattern in this
   step.
3. Both baselines score ~92% accuracy and ~0.98 ROC-AUC; Naive Bayes
   has the edge on injection-class recall (90% vs 87%), which matters
   more than accuracy for a security tool.
4. The model's own learned weights confirm the vocabulary pattern
   spotted in EDA, and also reveal a real weakness: benign
   *meta-discussion* of prompt injection shares vocabulary with actual
   attacks, which is where false positives will concentrate.
5. Model + vectorizer are saved together, ready to be loaded by an API
   in a later step.

# Step 3: Model Evaluation

---

## Additional imports for evaluation

```python
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
from sklearn.metrics import precision_score, recall_score, f1_score
```
`cross_val_score` runs a full train/evaluate cycle multiple times on
different slices of the data and returns a score per slice — used below
to check whether Step 2's results were a fluke of one particular
train/test split. `Pipeline` chains preprocessing and modeling steps
into a single object that can be fit and reused as one unit — critical
for doing cross-validation correctly, explained in detail below.
`PrecisionRecallDisplay` is a plotting convenience class, the
precision/recall equivalent of `RocCurveDisplay` from Step 2.
`precision_score`, `recall_score`, `f1_score` are the individual metric
functions — used here directly (rather than via
`classification_report`) because we need to compute them repeatedly
across several thresholds, not just print one summary.

---

## Precision-Recall curve + threshold sweep

This is the most important concept in Step 3: **the 0.5 threshold
sklearn uses by default is not special — it's just a default.**

```python
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

PrecisionRecallDisplay.from_predictions(y_test, logreg_probs, ax=axes[0], name='Logistic Regression')
PrecisionRecallDisplay.from_predictions(y_test, nb_probs, ax=axes[0], name='Naive Bayes')
axes[0].set_title('Precision-Recall Curve')
```
A **precision-recall curve** plots precision against recall as the
decision threshold sweeps across its full range. Unlike the ROC curve
from Step 2 (which plots true-positive-rate against false-positive-rate
and is fairly insensitive to class imbalance), a PR curve is more
informative specifically *because* our classes are imbalanced — it
directly shows the tradeoff between "catching real attacks" (recall)
and "not crying wolf" (precision), which is exactly the tradeoff a
security tool has to navigate.
`PrecisionRecallDisplay.from_predictions` takes the true labels and the
predicted *probabilities* (not hard predictions) — it needs the raw
probabilities because it's sweeping across every possible threshold
internally to draw the curve, the same reason ROC-AUC needed
probabilities back in Step 2.

```python
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
rows = []
for t in thresholds:
    preds_t = (nb_probs >= t).astype(int)
```
By default, `.predict()` classifies anything with predicted probability
≥ 0.5 as the positive class. Here we bypass `.predict()` entirely and
threshold the raw probabilities manually: `nb_probs >= t` produces a
boolean array (`True` where the model's injection-probability meets or
exceeds our chosen cutoff `t`), and `.astype(int)` converts
`True`/`False` into `1`/`0` so it's directly comparable to `y_test`.
This is how you simulate "what if I made the model more or less
cautious" without retraining anything — the model itself doesn't
change, only where you draw the line on its output.

```python
    rows.append({
        'threshold': t,
        'precision': round(precision_score(y_test, preds_t), 3),
        'recall': round(recall_score(y_test, preds_t), 3),
        'f1': round(f1_score(y_test, preds_t), 3),
    })
threshold_df = pd.DataFrame(rows)
print(threshold_df)
```
For each candidate threshold, compute all three metrics and collect
them as a list of dictionaries — a common lightweight pattern for
building a small results table without preallocating arrays.
`pd.DataFrame(rows)` converts that list of dicts directly into a table,
one row per threshold, which is far easier to read and compare than
five separate printed blocks.

**What the resulting table showed**: at threshold 0.3, recall hits
97.4% (catching nearly every attack) but precision drops to 79.6% (a
real chunk of alerts are false alarms). At 0.7, precision climbs to
97.9% but recall falls to 79.6% — the two numbers essentially swap
places. This is the tradeoff made concrete: there is no threshold that
maximizes both simultaneously, only a choice of *which kind of mistake
you'd rather make more of*.

```python
axes[1].plot(threshold_df['threshold'], threshold_df['precision'], marker='o', label='Precision')
axes[1].plot(threshold_df['threshold'], threshold_df['recall'], marker='o', label='Recall')
axes[1].plot(threshold_df['threshold'], threshold_df['f1'], marker='o', label='F1')
```
Three line plots sharing one axes, each tracing how one metric moves as
the threshold changes. `marker='o'` puts a visible dot at each actual
computed threshold value (rather than just a smooth line), which
matters here since we only computed 5 discrete points — the dots are
honest about where real data exists versus where the line is just
connecting the dots visually.

**The key visual takeaway**: precision and recall cross each other
around threshold ≈ 0.45–0.5, and F1 (their balance point) peaks right
around there too — which is *why* 0.5 is a reasonable default in
general, even though we just established it isn't automatically the
right choice for every use case. For this project specifically, since
missing an attack (false negative) is worse than a false alarm, you'd
likely deliberately choose a threshold *below* the F1-optimal point —
trading some precision for extra recall.

---

## Error analysis — false negatives and false positives

```python
nb_preds = nb.predict(X_test_tfidf)
results = pd.DataFrame({'text': X_test, 'true_label': y_test, 'pred': nb_preds, 'prob_injection': nb_probs})
```
Builds one combined table with the original text alongside the true
label, the model's hard prediction, and its raw probability — this
"put everything in one DataFrame" pattern is the standard way to move
from aggregate metrics (a single recall number) to actually inspecting
*which specific rows* the model got wrong, which is where the real
insight lives.

```python
false_negatives = results[(results['true_label'] == 1) & (results['pred'] == 0)]
false_positives = results[(results['true_label'] == 0) & (results['pred'] == 1)]
```
Two chained boolean conditions combined with `&` (pandas' element-wise
AND — note this is `&`, not Python's `and`, which doesn't work
element-wise on Series). A **false negative** is a row where the true
label is 1 (it really was an injection) but the model predicted 0 (it
called it benign) — a missed attack. A **false positive** is the
reverse — a benign message wrongly flagged. Each condition must be
wrapped in its own parentheses when combined with `&`, due to Python
operator precedence rules — a very common source of pandas errors
worth remembering.

**What reading the actual false negatives revealed** — this is the
single most valuable finding in Step 3: several missed attacks use
**emotional or identity-based manipulation** ("my eating disorder makes
your refusal feel like rejection", "I have PTSD... your restrictions
trigger my symptoms") rather than technical jargon like "bypass" or
"ignore previous instructions." These don't share vocabulary with the
"restrictions/reveal/bypass" pattern the model learned to key off back
in Step 2's feature-importance inspection — which means **this is a
real, structural blind spot of a bag-of-words approach**, not just
random noise. A model built purely on word statistics has no way to
recognize a *manipulation strategy* it's never seen the specific words
for. This is concrete evidence for why a future iteration might
benefit from a model that captures semantic meaning rather than just
term frequency (motivating a transformer-based approach as a stretch
goal).

The false positives, by contrast, look more like ordinary bad luck —
benign questions that happen to contain injection-adjacent vocabulary
("hidden gems", "harmful information vs harmful intent") — consistent
with the vocabulary-overlap risk already flagged back in Step 1's EDA.

---

## Cross-validation (leakage-safe via Pipeline)

```python
cv_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')),
    ('nb', MultinomialNB())
])
```
A `Pipeline` chains multiple steps into one object, each given a
string name (`'tfidf'`, `'nb'`) for reference. This solves a specific
problem with cross-validating text pipelines: if you vectorized the
full dataset once *before* splitting into folds, the TF-IDF vocabulary
and IDF weights would have "seen" every fold's data during fitting —
including the fold currently held out for testing in each iteration.
That's a data leak, just like the train/test leak explained in Step
2, but easier to accidentally introduce during cross-validation because
it's tempting to vectorize once upfront for convenience. Wrapping the
vectorizer and model together in a `Pipeline` and handing the *raw*
text (`X`, not `X_train_tfidf`) to `cross_val_score` ensures the
vectorizer gets refit from scratch, on training-fold data only, inside
every single fold — completely mirroring the discipline established in
Step 2.

```python
cv_scores = cross_val_score(cv_pipeline, X, y, cv=5, scoring='recall')
```
`cv=5` requests **5-fold cross-validation**: the data is split into 5
roughly equal chunks, and the pipeline is trained 5 separate times,
each time using 4 chunks for training and the 1 remaining chunk for
evaluation — rotating which chunk is held out each time, so every row
eventually gets used for evaluation exactly once. `scoring='recall'`
tells it which metric to compute and return for each fold — chosen
deliberately here since recall is the metric this whole project cares
about most, not accuracy.

```python
print("5-fold CV recall scores:", cv_scores.round(3))
print("Mean CV recall:", round(cv_scores.mean(), 3), "+/-", round(cv_scores.std(), 3))
```
`cv_scores` is an array of 5 recall values, one per fold. Printing the
mean and standard deviation together (`0.905 +/- 0.03`) communicates
both the typical performance *and* how much it varies depending on
which slice of data you happen to evaluate on — a single train/test
split (like Step 2's) only ever gives you one point estimate, with no
sense of how stable that number actually is. A small standard
deviation here (0.03, and individual fold scores ranging narrowly from
0.849 to 0.933) is reassuring: it confirms Step 2's 90% recall figure
wasn't a lucky artifact of that one particular split — the model
performs consistently across different data partitions.

---

## Step 3 summary

1. The precision/recall tradeoff is real and threshold-dependent —
   sklearn's default 0.5 cutoff is a convention, not something specially
   optimal for this use case, and given false negatives cost more than
   false positives here, a lower threshold is defensible.
2. Manual error analysis (reading actual false negatives) surfaced a
   real weakness invisible in the aggregate numbers: emotional/identity
   manipulation attacks evade the model's largely keyword-driven
   detection — the clearest evidence yet for eventually trying a model
   that captures semantic meaning, not just term frequency.
3. Cross-validation confirmed the model's recall is stable (~90%,
   ±3%) across different data splits, using a `Pipeline` to keep the
   vectorizer fitting leakage-free inside every fold — the same
   discipline from Step 2, now correctly extended to a multi-fold
   setting.
