# Practicing imbalanced learning

## Introduction

This file is a step by step tutorial to practice basics of imbalanced learning.
You will also find datasets and Jupyter notebooks with code that could help you
address the questions.

## About imbalance

We will consider the question of 2-class classification.
In general we consider or we suppose that both classes are equally represented
in the dataset.
What is imbalance? It is when one of the classes in less frequent than the
other.
In 2-class classification, if one class is <10% we have imbalance.

## More imbalance

Imbalance shows up all the time in real life, and in particular in industrial
use cases.
Predictive maintenance: some equipments have dysfunctions, but sometimes only
once every two years!
Quality control: by design, the proportion of samples that do not meet the
quality is very low.
The fields of anomaly detection and rare events prediction are linked to
imbalanced data.

- Finance: fraud detection datasets commonly have a fraud rate of ~1–2%
- Ad Serving: click prediction datasets have click-through-rate <1%.
- Industry: when will the next failure occur?
- Medical: classification of patients with rare condition
- Content moderation: detection of unsafe content

## Example of output for this tutorial

You may want to display your results in a table as such:

|ID|Classifier&ensp;name|Modifier|Sampling|Dataset&ensp;size|%&ensp;imbalance|Training&ensp;precision|Training&ensp;accuracy|Training&ensp;recall|Training&ensp;f1|Training&ensp;balanced&ensp;accuracy|Training&ensp;size|Testing&ensp;precision|Testing&ensp;accuracy|Testing&ensp;recall|Testing&ensp;f1|Testing&ensp;balanced&ensp;accuracy|Testing&ensp;size|
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|1|Random&ensp;Forest|N/A|N/A|1000|10|-|-|-|-|-|-|-|-|-|-|-|
|2|Random&ensp;Forest|class_weight|N/A|1000|10|-|-|-|-|-|-|-|-|-|-|-|
|3|Random&ensp;Forest|class_weight|Oversampling|1000|...|-|-|-|-|-|-|-|-|-|-|-|
|4|...|...|...|...|...|-|-|-|-|-|-|-|-|-|-|-|

## Datasets - synthetic data

If you need help: see notebook `00_imbalanced_synthetic.ipynb`

Let's create and plot several toy datasets that we will use for this course.
Let's try 2D/10D/20D data, two classes, 1-2-5-10-20% imbalance, size 10000.

Suggestions:

- Install imbalanced-learn:
- <https://imbalanced-learn.org/stable/install.html>
- Create a Jupyter notebook.
- Use `imbalanced learn` or `sklearn` to build and plot some imbalanced datasets
- `from sklearn.datasets import make_classification`
- `from imblearn.datasets import make_imbalance`

## Splitting

Let's split our datasets and keep an validation set on the side. It should be
stratified correctly.

Suggestions:

- In your notebook, create a validation set that we will call hold-out set for
- each of the synthetic datasets
- Use Stratify in `train_test_split`
- (We will also use Stratification in K fold validation later on)

## Baseline model

Then we can work on a baseline model, a simple classification algorithm such as
Random Forests.

sklearn doc: <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>

Suggestions:

- Write a Random Forest classifier for all datasets, it’s ok to keep the
default hyperparameters for now.
- If you like you can also write a Support Vector Machine (`SVC` in `sklearn`).

## Training the baseline models

Let's train it, and measure basic performance on our validation set with a
confusion matrix
We have TP, TN, FP, FN.

Suggestions:

- Train your baseline models and compute basic confusion matrix metrics on
train set and on hold-out set.

## Classic Metrics

We can compute the usual metrics stemming from a confusion matrix: precision,
accuracy, recall.

Let's see how it goes for all our datasets.
Typically precision and accuracy are very high but do not reflect at all what
we want.

You get a high accuracy just by predicting the majority class, but you fail to
capture the minority class, which is most often the point of the question.

sklearn doc: <https://scikit-learn.org/stable/modules/model_evaluation.html>

Wikipedia: <https://en.wikipedia.org/wiki/Confusion_matrix>

## Alternative metrics more suited to imbalance

Instead of the regular classification metrics, you can use more suited metrics.

F1 score is the harmonic mean of precision and recall.

`F-beta` score strikes a balance between precision and recall.
If we want to prioritize precision we can use `beta=0.5`
If we want to prioritize recall we can use `beta=2`

Balanced accuracy is also a better choice as it takes into account the relative
size of the classes.

Let's calculate it for all our datasets.

Suggestions:

- Compute all metrics for your baseline model!

## Tweaks to classification models to handle imbalance

Some classification models can behave a bit better than others with imbalanced
data, through the use of “class_weight” parameter (Decision Tree, Random Forest,
SVC)

The “balanced” mode uses the values of y to automatically adjust weights
inversely proportional to class frequencies in the input data as
`n_samples / (n_classes * np.bincount(y))`

sklearn doc: <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>

Suggestions:

- Try class_weight parameter in Random Forest
- See influence on the metrics

## Alternative classification models more suited to imbalance

Besides the metrics change you can also use different machine learning models,
that are more suited to imbalance.

For instance Balanced Random Forests is a variation on Random Forests that
deals with class imbalance natively.

For each iteration of RF, take a bootstrap of minority class, then take the
same number of observations in majority class

Reference: Chao Chen, Andy Liaw, Leo Breiman, and others. Using random forest
to learn imbalanced data. University of California, Berkeley, 110(1-12):24, 2004.

Imbalanced-learn ref: <https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedRandomForestClassifier.html>

Suggestions:

- Try balanced random forests from imbalanced-learn
- See influence on the metrics

If you are in a strongly imbalanced case, you can decide to use one-class
classification models such as isolation forests or single class svm.
You can also use anomaly detection techniques.

## Change the input data

Then, a powerful method to address class imbalance is to use undersampling and
oversampling techniques.
If the minority class has a sufficient number of representants, then you can
try undersampling the majority class at random.

Otherwise, you must find clever ways to oversample the minority class. You can
simply duplicate data.

You could also try to add noise to members of the minority class.

Suggestions:

- Use `imbalanced-learn` to randomly undersample and oversample.
- See influence on the metrics

## SMOTE

Finally, you could use SMOTE -
Synthetic Minority Oversampling TEchnique.

SMOTE allows you to interpolate between members of your minority class in
order to create additional data points.

For each sample in the minority class:
Compute its K-Nearest Neighbors (KNN).
Select one of them randomly.
Synthesize a new observation by linear interpolation.

Algorithm: <https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume16/chawla02a-html/node6.html>

Let's use SMOTE on our toy datasets and plot them.
Then let's learn on these augmented datasets and see how the test metrics
change.

Suggestions:

- Use `imbalanced-learn` to carry out SMOTE.
- Plot datasets.
- See influence on the metrics.

## Datasets - real data

If you need help: see notebook `01_imbalanced_public_dataset.ipynb`

There are several public datasets that exhibit imbalance, now that we went all
the way with synthetic data, we can use them too.

A few of them are available in imbalanced-learn directly: see <https://imbalanced-learn.org/stable/references/generated/imblearn.datasets.fetch_datasets.html>

Suggestions:

- Choose a couple of datasets in `imbalanced-learn`.
- Explore the data, formulate the question.
- Train models, look at metrics.

## SECOM dataset

If you need help: see notebook `02_imbalanced_secom.ipynb`

<https://archive.ics.uci.edu/ml/datasets/SECOM>

A complex modern semi-conductor manufacturing process is  under consistent
surveillance via the monitoring of signals/variables collected from sensors
and/or process measurement points. The dataset presented in this case represents
a selection of such measurements where each example represents a single
production entity with associated measured features and the labels represent a
simple pass/fail yield for in house line testing, where –1 corresponds to a
pass and 1 corresponds to a fail and the data time stamp is for that specific
test point.

Quality control question.

Suggestions:

- Download the data.
- Explore the data, formulate the question.
- Choose one model, metric, and data augmentation technique.
- Train your model with cross-validation, look at metrics.

## NASA Turbofan dataset

If you need help: see notebook `03_imbalanced_turbofan.ipynb`

<https://www.kaggle.com/datasets/behrad3d/nasa-cmaps>

Use only FD001.

Dataset presents Run-to-Failure simulated data from turbo fan jet engines. It
consists of multiple multivariate time series. Each time series is from a
different engine. Each engine starts with different degrees of initial wear and
manufacturing variation which is unknown to the user.
The engine is operating normally at the start of each time series, and develops
a fault at some point during the series. In the dataset, the fault grows in
magnitude until system failure.

We want to predict whether the unit will fail within the next 5 cycles.

Suggestions:

- Download the data.
- Explore the data, formulate the question.
- Choose one model, metric, and data augmentation technique.
- Train your model with cross-validation, look at metrics.

Tips:

- You will need custom stratification strategies.
- Unit number is useless (except for stratification).
- You can create features that exploit time.

## Conclusion and perspectives

Data imbalance is frequent and you can deal with it in several ways.
You can:

- change your data,
- use appropriate models,
- use appropriate metrics.

More about SMOTE:
There are other approaches that can be used for regression questions, in
particular one is SMOTER.
At Fieldbox we also studied an algorithm for sequence-to-sequence questions,
which is called [SMOTEST](https://github.com/fieldboxai/predict-rare-events-smotest).
SMOTE has been discussed in a recent paper on medical trials and seem to be
counterproductive sometimes.

Ideas of complementary work:

- Find other datasets and work with them
- Study SMOTER and implement it in Python
- Study [this paper](https://academic.oup.com/jamia/article/29/9/1525/6605096?login=false),
that tempers SMOTE:
van den Goorbergh, Ruben, et al. "The harm of class imbalance corrections for
risk prediction models: illustration and simulation using logistic regression."
Journal of the American Medical Informatics Association (2022).
- Watch [this video](https://www.youtube.com/watch?v=6YnhoCfArQo) of Guillaume
Lemaitre at [euroscipy2023](https://pretalx.com/euroscipy-2023/talk/GYYTCH/)
about some advanced `scikit-learn` features, some of which deal with imbalance.

## Notes

Used in 2022, 2023 and 2024 with AI engineering students in Bordeaux.
