# **Iris Species Classification Using Machine Learning**

## **Author**
Touseef Asif

---

## **Project Overview**

The Iris dataset is one of the most well-known datasets in the machine learning community. This project involves building a machine learning model to classify three different species of Iris flowers — **Iris-setosa**, **Iris-versicolor**, and **Iris-virginica** — based on their petal and sepal measurements.

In this project, two machine learning algorithms, **Logistic Regression** and **K-Nearest Neighbors (KNN)**, are used to build and evaluate classification models. The performance of both models is compared, and predictions are made on new sample data.

---

## **Dataset Information**

The dataset used for this project is the classic **Iris dataset**, which contains:
- **150 samples**.
- **4 features**: 
  - Sepal Length (cm)
  - Sepal Width (cm)
  - Petal Length (cm)
  - Petal Width (cm)
- **3 target classes**:
  - Iris-setosa
  - Iris-versicolor
  - Iris-virginica

The dataset file is named `Iris.csv`.

---

## **Technologies and Libraries Used**

The project is implemented in **Python**, and the following libraries are used:
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations.
- **Matplotlib & Seaborn**: For data visualization.
- **Scikit-learn**: For preprocessing, model training, and evaluation.

---

## **Project Workflow**

### **1. Data Loading**
- The dataset is loaded from the CSV file using `pandas.read_csv()`.
- Initial exploration of the dataset is performed to understand its structure and features.

### **2. Exploratory Data Analysis (EDA)**
- **Dataset Summary**: The dataset is summarized to check for missing values, data types, and basic statistics.
- **Visualizations**:
  - **Pairplot**: Visualize the relationship between features, grouped by species.
  - **Heatmap**: Display the correlation between numerical features.

### **3. Data Preprocessing**
- **Encoding Target Variable**: The species column is encoded into numeric labels using `LabelEncoder`.
- **Splitting the Dataset**: The dataset is split into training and testing sets (80% training, 20% testing).
- **Feature Scaling**: Features are scaled using `StandardScaler` for better model performance.

### **4. Model Building**
- **Logistic Regression**:
  - A logistic regression model is trained on the training set.
  - The model's accuracy, confusion matrix, and classification report are evaluated.
- **K-Nearest Neighbors (KNN)**:
  - A KNN classifier (with `k=5`) is trained on the training set.
  - The model's accuracy, confusion matrix, and classification report are evaluated.

### **5. Model Evaluation**
- The accuracy of both models is compared using a bar plot.

### **6. Predictions**
- Predictions are made on new sample data using the trained KNN model.

---

## **Results and Outputs**

### **Model Evaluation**
The models achieved the following accuracies on the testing set:
- **Logistic Regression Accuracy**: 97.0%  
- **KNN Accuracy**: 98.3%

### **Prediction on New Data**
A sample input with the following measurements:
```plaintext
SepalLengthCm: 5.1
SepalWidthCm: 3.5
PetalLengthCm: 1.4
PetalWidthCm: 0.2

Predicted Species: Iris-setosa
