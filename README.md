# Iris Species Classification

This project demonstrates the use of machine learning with the Random Forest Classifier to classify iris flowers into three species based on their sepal and petal measurements. Utilizing the famous Iris dataset, this project encompasses data preprocessing, model training, and evaluation to accurately predict the species of iris flowers.

## Project Overview

The Iris dataset is a foundational dataset for classification problems in the field of machine learning and data science. This project aims to preprocess the data, train a Random Forest Classifier, and evaluate its performance through accuracy metrics, a confusion matrix, and a classification report.

## Getting Started

### Prerequisites

This project requires Python and the following Python libraries:

- Pandas
- Scikit-learn
- Seaborn
- Matplotlib

You can install these libraries using pip:

```sh
pip install pandas scikit-learn seaborn matplotlib
```

### Dataset
The Iris dataset includes 150 instances of iris flowers, each with four features (sepal length, sepal width, petal length, petal width) and a target variable representing the species (setosa, versicolor, virginica).

## Analysis Steps

### Data Preprocessing: 
The dataset is split into features (X) and the target variable (y), followed by splitting into training and testing sets.

### Model Training: 
A Random Forest Classifier is initialized and trained with the training data.

### Model Evaluation: 
The trained model is used to predict the species of iris flowers in the test set. Model performance is evaluated using accuracy, a confusion matrix, and a classification report.

### Running the Analysis
Download the Iris dataset and update the file_path variable in the script to point to the location of the IRIS.csv file on your machine.

Execute the Python script. It will preprocess the data, train the Random Forest Classifier, and output the model's accuracy, confusion matrix, and classification report.

### Results
The accuracy score indicates the model's overall performance in classifying the iris species correctly.
The confusion matrix provides insights into the true positives, true negatives, false positives, and false negatives.
The classification report includes precision, recall, f1-score, and support for each class.

### Conclusion
This project illustrates the effectiveness of the Random Forest Classifier in classifying the species of iris flowers. Through careful data preprocessing, model training, and evaluation, we can achieve high accuracy in predicting the correct species based on sepal and petal measurements.
