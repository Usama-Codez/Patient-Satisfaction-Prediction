# 🏥 Patient Satisfaction Prediction using Machine Learning

Patient Satisfaction Prediction is a machine learning project that predicts **patient satisfaction** levels using doctor related metrics. Designed with efficiency, clarity, and model benchmarking in mind, complete with automated evaluation.

---

## 📊 Project Overview

This project explores how doctor experience, fees, wait times, and other metrics impact **patient satisfaction**. It applies various machine learning techniques to:
- 📌 Preprocess and normalize raw data
- 🤖 Train and evaluate multiple models (manually + via LazyPredict)
- 📈 Benchmark performances
- 📁 Save results for further analysis

---

## 🧠 Features

- ✅ Handles missing values automatically
- 📐 Normalizes numerical data using `MinMaxScaler`
- ⚖️ Balances dataset using **SMOTE**
- 🧪 Trains multiple classifiers including:
  - Logistic Regression
  - Random Forest
  - Decision Tree
  - AdaBoost
  - KNN
  - SVM
  - Naive Bayes
  - XGBoost
- 🤖 Uses `LazyPredict` to automatically benchmark many models
- 📊 Saves model performance comparisons as CSV
- 🎯 Voting Classifier for accuracy boosting
- 📦 Cleanly modular and well-commented code

---

## 🗂️ Dataset

- Location: `./dataset/doctors_dataset.csv`
- Contains:
  - Experience (Years)
  - Total Reviews
  - Satisfaction Rate (%)
  - Avg Time to Patients
  - Wait Time
  - Fee (PKR)
  - Target label (Satisfaction: Satisfied / Unsatisfied)

---

## 🧪 Algorithms & Tools Used

| Type              | Tools/Libraries                        |
|-------------------|----------------------------------------|
| Programming       | Python 3.11+                           |
| ML Frameworks     | `scikit-learn`, `xgboost`, `lazypredict`, `imblearn` |
| Visualization     | `seaborn`, `matplotlib`                |
| Data              | `pandas`, `numpy`                      |
| IDE               | Jupyter Notebook / VS Code             |

---

## 🔄 How to Run the Project

### 1. Set up the environment

```bash
# Clone the repo
git clone https://github.com/Usama-Codez/Patient-Satisfaction-Prediction.git
cd Patient-Satisfaction-Prediction
```
Create & activate a virtual environment
```bash
python -m venv newenv
newenv\Scripts\activate
```

Install dependencies
```bash
pip install -r requirements.txt
```

### 2. 🧠 Run the notebook

You can Use any of the following IDEs:
- Jupyter Notebook
- Jupyter Lab
- Google Colab
Make sure to select the correct environment when running your notebook.

📉 📈 Confusion Matrix, ROC curve and Evaluations:

### Correlation Heat Map:

### Confusion Matrix Logistic Regression model:
![image](https://github.com/user-attachments/assets/d4b9ca1b-34c4-47f6-919f-7bb3c5c28cc3)

### ROC Curve Logistic Regression model:
![image](https://github.com/user-attachments/assets/9dc60b85-7121-41ac-8e23-216ed3989bd3)

### Evaluating LogisticRegression
Accuracy: 58.76%
Precision: 0.95
Recall: 0.59
F1 Score: 0.73
ROC-AUC: 0.5667822806377023

### Confusion Matrix Random Forest Classifier:
![image](https://github.com/user-attachments/assets/c7d2a877-d886-4797-a234-da376fa59b6d)

### ROC Curve Random Forest Classifier:
![image](https://github.com/user-attachments/assets/b47c15fc-ffd2-46b2-ad8b-66e0a06e87e0)

### Evaluating Random Forest Classifier
Accuracy: 90.58%
Precision: 0.95
Recall: 0.95
F1 Score: 0.95
ROC-AUC: 0.735426554703663

### Confusion Matrix XGBoost:
![image](https://github.com/user-attachments/assets/bd891d1c-52dc-4e8e-b9e0-d1e8c704b33d)


### ROC Curve XGBoost:
![image](https://github.com/user-attachments/assets/b86b9bd4-7215-4f63-a561-9ba4d4983033)

### Evaluating XGBoost
Accuracy: 90.96%
Precision: 0.94
Recall: 0.96
F1 Score: 0.95
ROC-AUC: 0.7725142996227334

### Confusion Matrix SVM:
![image](https://github.com/user-attachments/assets/020a9225-6ce0-416f-8916-5c23439b494d)

### ROC Curve SVM:
![image](https://github.com/user-attachments/assets/3bd50380-1709-422d-a298-4f10ef931010)

### Evaluating SVM
Accuracy: 72.32%
Precision: 0.95
Recall: 0.75
F1 Score: 0.84
ROC-AUC: 0.6031702567847147

### Confusion Matrix KNN:
![image](https://github.com/user-attachments/assets/90b232e9-4c0b-408e-80b6-5c55abbd417f)

### ROC Curve KNN:
![image](https://github.com/user-attachments/assets/a19d48b7-c54b-484b-aaee-9aeee267a18e)

### Evaluating KNN
Accuracy: 76.27%
Precision: 0.94
Recall: 0.80
F1 Score: 0.86
ROC-AUC: 0.5284166970913959

### Confusion Matrix Decision Tree:
![image](https://github.com/user-attachments/assets/ed0b8254-47f4-4bef-bec7-842790785659)

### ROC Curve Decision Tree:
![image](https://github.com/user-attachments/assets/abc3fc56-bea6-4d13-9b28-d46a6d27d34c)

### Evaluating DecisionTree
Accuracy: 79.66%
Precision: 0.95
Recall: 0.83
F1 Score: 0.88
ROC-AUC: 0.6035049288061336

### Confusion Matrix NaiveBayes:
![image](https://github.com/user-attachments/assets/961257e7-1edf-4c28-94a9-683d63de3909)

### ROC Curve NaiveBayes:
![image](https://github.com/user-attachments/assets/5de59ba6-5d07-4bca-ad3d-76855558f947)

### Evaluating NaiveBayes
Accuracy: 63.09%
Precision: 0.95
Recall: 0.64
F1 Score: 0.76
ROC-AUC: 0.5259827187538031

### Confusion Matrix AdaBoost:
![image](https://github.com/user-attachments/assets/38a86eed-f978-45cc-a7e0-22152a6e4415)

### ROC Curve AdaBoost:
![image](https://github.com/user-attachments/assets/4b0ceb63-3c78-4813-93c5-806ec9fc0f72)

### Evaluating AdaBoost
Accuracy: 77.21%
Precision: 0.96
Recall: 0.79
F1 Score: 0.87
ROC-AUC: 0.7437325057806986

### Testset evaluations:
Fitting 3 folds for each of 108 candidates, totalling 324 fits
Best Parameters: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
Best Cross-Validation Accuracy: 0.9363507779349364


### Author: Usama Akram

### 📌 License
This project is licensed under the MIT License.

