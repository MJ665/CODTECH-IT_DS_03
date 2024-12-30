
---

# Wine Classification using Various Machine Learning Models

## Overview
This project demonstrates how to apply different classification algorithms on the Wine dataset to predict wine categories based on chemical attributes. The models used include:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)

The workflow involves:
- Loading and preprocessing the Wine dataset.
- Training multiple machine learning models.
- Evaluating model performance using metrics such as Accuracy, Precision, Recall, F1 Score, and Confusion Matrix.
- Visualizing model comparison and confusion matrices.
- Providing a CLI interface for user input to predict wine categories.

## Requirements
- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install the required libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Dataset
The **Wine dataset** is a well-known dataset for classification tasks, containing 178 samples of wines derived from three different cultivars. The dataset has 13 features representing chemical properties of the wine and a target variable representing the wine cultivar.

- Features include: Alcohol, Malic Acid, Ash, Alcalinity of Ash, Magnesium, etc.
- The target variable (`target`) has three classes corresponding to three wine cultivars.

## Steps in the Code

### 1. Load and Explore the Dataset
The dataset is loaded using the `load_wine()` function from `sklearn.datasets`, and it's converted into a pandas DataFrame. Features and the target variable are separated.

### 2. Preprocessing the Data
- The data is split into training and testing sets using `train_test_split`.
- Feature scaling is applied using `StandardScaler` to standardize the features for better model performance.

### 3. Train and Evaluate Models
- Four models are trained: Logistic Regression, Decision Tree, Random Forest, and Support Vector Machine (SVM).
- Model performance is evaluated using:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1 Score**
  - **Confusion Matrix**

### 4. Visualizing Model Performance
- A bar plot is created for comparing the models' performance based on accuracy, precision, recall, and F1 score.
- A confusion matrix heatmap is plotted for the best-performing model (based on F1 score).

### 5. CLI for User Input and Prediction
A command-line interface (CLI) allows users to input values for the wine features and make predictions. The model used for predictions is the one with the best F1 score. Users can input wine feature values as comma-separated values, and the program will output the predicted wine category.

### Example Input
- **Input**: `13.2, 2.8, 2.1, 18.4, 96.0, 0.60, 2.25, 20.0, 0.30, 1.8, 3.1, 0.85, 2.5`
- **Output**: Predicted wine category.

## Results
After training the models, the following results are displayed:
- Performance metrics (Accuracy, Precision, Recall, F1 Score) for each model.
- Confusion matrix for the best-performing model.

## Usage
1. Run the script in a Python environment.
2. The script will display the performance of each model.
3. The best model (based on F1 score) will be used for predictions.
4. You can enter wine feature values in the CLI to predict the wine class.

## Conclusion
This project demonstrates the application of different classification algorithms for predicting wine categories. The model performance is compared, and the best model is selected for making predictions. The CLI offers an interactive way for users to predict wine categories based on new input values.

---

