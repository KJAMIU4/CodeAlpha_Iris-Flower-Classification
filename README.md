Iris Classifier – README
Project Overview

This project implements a Decision Tree Classifier to predict the species of iris flowers based on the classic Iris dataset. The workflow covers data loading, preprocessing, model training, evaluation, and performance reporting. The notebook provides a simple, beginner-friendly example of supervised machine learning using Python and scikit-learn.

Dataset

The model uses the Iris dataset, which contains 150 samples with the following features:

SepalLengthCm

SepalWidthCm

PetalLengthCm

PetalWidthCm

Species (Target variable)

Each sample belongs to one of three species:

Iris-setosa

Iris-versicolor

Iris-virginica

The dataset is loaded from:

C:\Desktop\PROJECT\Iris.csv


If you clone or reuse this project, ensure that the dataset path is correctly updated.

Project Workflow
1. Import Dependencies

The notebook uses the following libraries:

pandas

numpy

scikit-learn (DecisionTreeClassifier, train_test_split, accuracy_score, classification_report)

2. Load and Explore Dataset

The dataset is imported into a DataFrame to inspect structure, columns, and general data quality.

3. Prepare Features and Target

Features (X): All columns except Species

Target (y): Species column

4. Split Data

Data is divided into:

80% Training set

20% Test set

Using:

train_test_split(X, y, test_size=0.2, random_state=42)

5. Train the Model

A DecisionTreeClassifier is initialized and trained using the training data.

6. Make Predictions

The model predicts species labels for the test set.

7. Evaluate the Model

Performance metrics include:

Accuracy Score

Precision, Recall, F1-score (via classification report)

Results

The notebook prints:

Overall accuracy

Detailed classification report showing metrics for each iris species

This helps assess how well the decision tree model generalizes to unseen data.

Requirements
Python Libraries

Install the required dependencies:

pip install pandas numpy scikit-learn

File Requirement

Ensure the file Iris.csv is placed in the correct directory or update the path in the notebook.
