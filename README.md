# Deep Neural Network (DNN) Project README

## Project Overview

This project demonstrates the creation and training of a Deep Neural Network (DNN) for classification using Python and the TensorFlow library. The goal of the project is to build a DNN model that can accurately classify banknote authenticity based on certain features. To achieve this, the project covers various stages, including data visualization, data preprocessing, model construction, training, and evaluation.

## Prerequisites

Before running the code in this project, make sure you have the following prerequisites installed:

- Python (3.x recommended)
- Jupyter Notebook or any Python IDE
- Required Python libraries (pandas, numpy, seaborn, matplotlib, scikit-learn, TensorFlow)

You can install the required libraries using the following command:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn tensorflow
```

## Project Structure

- `DNN_Project.ipynb`: Jupyter Notebook containing the code and explanations for the DNN analysis.
- `bank_note_data.csv`: CSV file containing the banknote authenticity dataset.

## Installation

1. Clone or download this project repository to your local machine.
2. Open the Jupyter Notebook (`DNN_Project.ipynb`) using your preferred Python IDE or Jupyter Notebook itself.
3. Ensure you have the dataset file named `bank_note_data.csv` in the same directory as the notebook.

## Usage

1. Open the Jupyter Notebook and run each cell step by step to follow the DNN analysis process.
2. The notebook contains detailed comments and explanations for each code cell to help you understand the workflow.

## Data Visualization

The project starts with data visualization:

- Loading the banknote authenticity dataset and exploring its structure.
- Visualizing data patterns, such as class distribution and pairwise relationships using pair plots.

## Data Preprocessing

The main part of the project involves data preprocessing:

- Scaling the features using StandardScaler from scikit-learn.
- Splitting the data into training and testing sets using `train_test_split` from scikit-learn.

## Deep Neural Network Model

The project creates a Deep Neural Network model using TensorFlow:

- Creating a Sequential model with input and hidden layers.
- Configuring the model with appropriate activation functions, loss function, and optimizer.
- Training the model using the training data with a specified number of epochs and batch size.

## Model Evaluation

The project evaluates the performance of the DNN model:

- Making predictions on the test data.
- Evaluating the model using a confusion matrix and classification report to assess accuracy, precision, recall, and F1-score.

## Comparison with Random Forest Classifier

As a comparison, the project also uses a Random Forest Classifier:

- Training a Random Forest Classifier from scikit-learn with 200 estimators.
- Evaluating the Random Forest model's performance using the same metrics as the DNN model.

## Contributing

If you want to contribute to this project, feel free to fork the repository, make changes, and create a pull request. We welcome any contributions or improvements.

---

Feel free to reach out if you have any questions or need further assistance with this project. Happy coding!
