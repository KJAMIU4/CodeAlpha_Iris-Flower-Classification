🌸 Iris Classifier

A simple machine learning project that builds a Decision Tree Classifier to predict iris flower species using the classic Iris dataset.
The notebook walks through data loading, preprocessing, model training, evaluation, and interpretation of results — ideal for beginners exploring supervised learning with scikit-learn.

📁 Project Structure
Iris Classifier.ipynb   # Main notebook containing the entire workflow
Iris Classifier.py      # Executable python code
Iris.csv                # Dataset (ensure path is correctly set)
README.md               # Project documentation

📊 Dataset Overview

The project uses the well-known Iris dataset, which contains:

150 samples

4 numerical features:

SepalLengthCm

SepalWidthCm

PetalLengthCm

PetalWidthCm

3 target classes (Species):

Iris-setosa

Iris-versicolor

Iris-virginica

Make sure your file path to Iris.csv is correctly set in the notebook.

🚀 Workflow Summary
1. Import Libraries

This project uses:

pandas

numpy

scikit-learn
(DecisionTreeClassifier, train_test_split, accuracy_score, classification_report)

2. Load Dataset

The CSV file is imported into a DataFrame for inspection and processing.

3. Feature Engineering

X → All columns except Species

y → Species

4. Split Data

Training: 80%
Testing: 20%
Using scikit-learn’s train_test_split.

5. Train Model

A Decision Tree Classifier is fitted on the training data.

6. Model Prediction

Predictions are made on the test set.

7. Model Evaluation

Performance is measured using:

Accuracy Score

Classification Report (Precision, Recall, F1-score)

🧪 Results

The notebook prints:

Overall model accuracy

A detailed breakdown of classifier performance across all three species

Decision trees generally perform well on this dataset, making it an excellent introduction to classification problems.

🛠️ Installation

Install required libraries:

pip install pandas numpy scikit-learn

▶️ How to Run

Clone this repository.

Open the notebook in Jupyter Notebook, JupyterLab, or VS Code.

Ensure dataset path is valid.

Run all cells sequentially.

📌 Future Enhancements

Potential improvements include:

Adding data visualizations (pair plots, correlation heatmap)

Trying more ML models:
SVM, KNN, Random Forest, Logistic Regression

Hyperparameter tuning with GridSearchCV

Deploying as a Streamlit or Flask app

Saving the trained model for production use

📄 License

This project is open-source and free to modify or extend.
