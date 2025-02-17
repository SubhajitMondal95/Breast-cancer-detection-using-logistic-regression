# Breast-cancer-detection-using-logistic-regression
This Python project demonstrates a basic machine learning pipeline for predicting breast cancer diagnosis (benign or malignant) using the Wisconsin Breast Cancer dataset.  The code implements both Logistic Regression and K-Nearest Neighbors (KNN) classification models, allowing for comparison of their performance.
## Overview

This project covers the fundamental steps involved in a typical machine learning workflow, including data loading, preprocessing, model training, evaluation, and visualization. It provides a clear and concise example of how to build a classification model for a binary classification problem.

## Features

* **Data Loading and Preprocessing:** Loads the Wisconsin Breast Cancer dataset from a CSV file (`breast_cancer.csv`).  Handles label encoding to convert categorical target variables into numerical representations.
* **Train-Test Split:** Divides the dataset into training and testing sets to evaluate model performance on unseen data.
* **Model Training:** Trains either a Logistic Regression or a K-Nearest Neighbors (KNN) classifier.  The code includes implementations for both.  *Note: You can easily switch between the models by commenting/uncommenting the relevant code blocks.*
* **Model Evaluation:** Evaluates the trained model using a confusion matrix, k-fold cross-validation, ROC curve, and accuracy comparison between training and testing sets.
* **Visualization:** Generates visualizations of the confusion matrix, ROC curve, and train/test accuracy comparison using `matplotlib` and `seaborn`.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

## Dataset

The project uses the Wisconsin Breast Cancer dataset, which is assumed to be in a CSV file named `breast_cancer.csv`.  This file should be placed in the same directory as the Python script.  The dataset should contain features relevant to breast cancer diagnosis and a target variable indicating the diagnosis (benign or malignant).  *Ensure the target variable is encoded as 2 and 4, which the code then maps to 0 and 1.*

## How to Run

1.  Clone the repository: `git clone https://github.com/YOUR_USERNAME/Breast-Cancer-Prediction.git` (Replace with your repository URL)
2.  Install the required libraries: `pip install pandas numpy scikit-learn matplotlib seaborn`
3.  Place the `breast_cancer.csv` file in the same directory as the Python script.
4.  Run the script: `python your_script_name.py` (Replace `your_script_name.py` with the name of your Python file).

## Model Selection

By default, the code runs Logistic Regression.  To switch to KNN, comment out the Logistic Regression code block and uncomment the KNN code block.  *Remember to scale your data if you are using KNN.*

## Evaluation Metrics

The project uses the following evaluation metrics:

* **Confusion Matrix:**  Visualizes the model's performance in terms of true positives, true negatives, false positives, and false negatives.
* **K-Fold Cross-Validation:**  Provides a more robust estimate of the model's performance by averaging the accuracy across multiple folds of the data.
* **ROC Curve and AUC:**  Illustrates the trade-off between true positive rate and false positive rate and provides a measure of the model's ability to distinguish between classes.
* **Train/Test Accuracy:**  Compares the model's accuracy on the training and testing sets to assess potential overfitting.

## Future Enhancements (Optional)

* **Feature Engineering:** Explore techniques to create new features from the existing ones to potentially improve model performance.
* **Model Tuning:** Implement hyperparameter tuning using techniques like GridSearchCV or RandomizedSearchCV to optimize the model's parameters.
* **Advanced Classification Models:**  Experiment with other classification algorithms, such as Support Vector Machines, Random Forests, or Gradient Boosting.
* **Deployment:** Deploy the trained model as a web application or using a framework like Flask or Streamlit.
