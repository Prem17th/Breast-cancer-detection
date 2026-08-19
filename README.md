# Breast-cancer-detection

A simple machine learning project for classifying breast cancer tumors as **Malignant** or **Benign** using the **Support Vector Machine (SVM)** algorithm.

Basically, this program uses the **Breast Cancer dataset** available in the Scikit-learn library. The dataset contains different measurements of breast cancer cells, which are used to predict whether a tumor is malignant or benign.

Dataset :-
The dataset contains several features related to breast cancer cells, such as:
* Mean radius
* Mean texture
* Mean perimeter
* Mean area
* Mean smoothness
* Mean compactness
* Mean concavity
* Mean symmetry
* And other cell measurements

The target contains two classes:
1. **Malignant**
2. **Benign**

Libraries Used :-
First, we import the required libraries:
* **NumPy** – for numerical operations
* **Pandas** – for creating and handling the dataset
* **Scikit-learn** – for machine learning
* **Matplotlib** – for displaying the confusion matrix

Loading the Dataset :-
We use the built-in `load_breast_cancer()` function from Scikit-learn to load the dataset.
```python
data = load_breast_cancer()
X = data.data
y = data.target
```

Here:
* `X` contains the input features.
* `y` contains the target/output values.

We then create a Pandas DataFrame to display the dataset in a table format.

Splitting the Dataset :-
The dataset is divided into two parts:
* **80% → Training data**
* **20% → Testing data**

This is done using `train_test_split()`.

The model learns from the training data and is then tested using the testing data.

Feature Scaling :-
Before training the SVM model, we use **StandardScaler**.
```python
scaler = StandardScaler()
```

Feature scaling is important because the dataset contains features with different ranges.
StandardScaler converts the features into a similar scale so that the SVM model can work more effectively.

SVM Model :-
We use the **Support Vector Machine (SVM)** algorithm for classification.
```python
svm_model = SVC(kernel='linear')
```

Here, we use a **linear kernel**.

The model is then trained using:
```python
svm_model.fit(X_train, y_train)
```

Prediction :-
After training the model, we use the test data to predict the classes:
```python
y_pred = svm_model.predict(X_test)
```

The predicted values are then compared with the actual values.

Model Accuracy :-
We calculate the accuracy of the model using:
```python
accuracy_score(y_test, y_pred)
```

This tells us how many predictions were correctly classified by the SVM model.


Classification Report :-
The program also generates a classification report containing:
* **Precision**
* **Recall**
* **F1-score**
* **Support**

This gives us more detailed information about the performance of the classification model.


Confusion Matrix :-
Finally, we create a **Confusion Matrix** using Matplotlib.
The confusion matrix shows:
* Correct predictions
* Incorrect predictions
* Actual classes
* Predicted classes

It helps us understand how well the SVM model classified malignant and benign tumors.

Conclusion :-
In this project, we used the **Breast Cancer dataset** and the **Support Vector Machine (SVM)** algorithm to classify tumors into malignant and benign categories.

The project demonstrates the basic machine learning workflow:
**Dataset → Data Preparation → Train/Test Split → Feature Scaling → SVM Training → Prediction → Evaluation → Confusion Matrix**
