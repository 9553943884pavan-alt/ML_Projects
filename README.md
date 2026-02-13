# Expected CTC Prediction Model

## 📋 Project Overview

This project develops a machine learning model to predict the Expected Cost-to-Company (CTC) for prospective employees at Company X. The goal is to automate salary determination while eliminating discrimination and ensuring fair compensation for employees with similar profiles.

## 🎯 Business Problem

The Human Resources department at Company X needs to maintain consistent and fair salary ranges for employees with similar profiles. Currently, salary decisions involve considerable human judgment based on factors like experience, skills, and abilities evaluated during interviews. This manual process can inadvertently introduce bias and discrimination in compensation decisions.

## 💡 Project Objective

Build a robust, data-driven model that:
- Automatically determines appropriate salary offers for selected candidates
- Minimizes manual judgment in the salary decision process
- Eliminates discrimination among employees with similar profiles
- Ensures fair and consistent compensation practices

## 🔬 Methodology & Approach

### 1. **Data Preprocessing**
- Exploratory Data Analysis (EDA) to understand data distribution
- Missing value analysis and treatment
- Feature selection based on importance
- Data cleaning and transformation

### 2. **Feature Engineering**
- Categorical feature encoding (OneHot and Ordinal encoding)
- Numerical feature scaling (MinMaxScaler, StandardScaler)
- Feature selection using domain knowledge
- Pipeline creation for preprocessing automation

### 3. **Model Development**
The primary model used is **CatBoostRegressor**, chosen for its:
- Superior handling of categorical features
- Robust performance with minimal hyperparameter tuning
- Built-in regularization to prevent overfitting
- Ability to capture complex non-linear relationships

**Key Features Handled:**
- Categorical: Department, Role, Industry
- Numerical: Experience, skills metrics, and other performance indicators

### 4. **Hyperparameter Optimization**
- GridSearchCV for systematic hyperparameter tuning
- Parameters optimized:
  - `iterations`: [100, 300, 500]
  - `depth`: [5, 7, 10]
  - `learning_rate`: [0.05, 0.1, 0.3]

### 5. **Model Evaluation**
The model is evaluated using multiple regression metrics:
- **R² Score**: Measures variance explained by the model
- **Mean Absolute Error (MAE)**: Average absolute prediction error
- **Mean Squared Error (MSE)**: Penalizes larger errors
- **Median Absolute Error**: Robust metric less affected by outliers

### 6. **Fairness & Bias Detection**
- **Fairlearn** library integration for bias assessment
- MetricFrame analysis to evaluate model fairness across protected attributes
- Ensuring predictions are equitable across different demographic groups

### 7. **Model Interpretability**
- **SHAP (SHapley Additive exPlanations)** for model explainability
- Understanding feature importance and contribution to predictions
- Transparency in salary determination decisions

## 🛠️ Technologies & Libraries

### Core Libraries
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **matplotlib & seaborn**: Data visualization

### Machine Learning
- **scikit-learn**: 
  - Model selection and evaluation
  - Preprocessing pipelines
  - Metrics calculation
- **CatBoost**: Primary regression model
- **scipy**: Statistical analysis

### Fairness & Interpretability
- **Fairlearn**: Bias detection and fairness metrics
- **SHAP**: Model interpretability and feature importance

## 📊 Model Pipeline

```
Raw Data
    ↓
Data Preprocessing
    ├── Numerical Features → StandardScaler/MinMaxScaler
    └── Categorical Features → OneHotEncoder/OrdinalEncoder
    ↓
Feature Engineering
    ↓
CatBoostRegressor Model
    ├── Hyperparameter Tuning (GridSearchCV)
    └── Cross-validation
    ↓
Model Evaluation
    ├── Performance Metrics (R², MAE, MSE)
    ├── Fairness Analysis (Fairlearn)
    └── Interpretability (SHAP)
    ↓
Predicted CTC
```

## 🎯 Key Features

1. **Automated Salary Prediction**: Eliminates manual bias in salary determination
2. **Fair & Unbiased**: Uses fairness metrics to ensure equitable predictions
3. **Interpretable**: SHAP values provide transparency in model decisions
4. **Robust**: CatBoost handles categorical features efficiently
5. **Scalable**: Pipeline architecture allows easy retraining with new data

## 📈 Results & Performance

The model is evaluated on both training and test datasets with metrics including:
- Training R² Score
- Test R² Score
- MAE, MSE, and Median Absolute Error
- Fairness metrics across protected attributes

## 🚀 Usage

### Prerequisites
```bash
pip install catboost
pip install shap
pip install fairlearn
pip install pandas numpy scikit-learn matplotlib seaborn scipy
```

### Running the Model
1. Load the dataset (`expected_ctc.csv`)
2. Run preprocessing pipeline
3. Train CatBoostRegressor with optimized hyperparameters
4. Evaluate model performance
5. Generate predictions for new candidates

## 📁 Project Structure

```
├── Expected_CTC_Prediction_Model.ipynb   # Main notebook
├── expected_ctc.csv                       # Dataset
├── README.md                              # Project documentation
└── requirements.txt                       # Dependencies
```

## 🔍 Model Insights

The model uses:
- **Multiple categorical features**: Department, Role, Industry
- **Numerical features**: Experience, skills, and performance metrics
- **Advanced preprocessing**: Column transformers with separate pipelines
- **Fairness checks**: Ensuring no discrimination across demographics

## 🎓 Learnings & Impact

This project demonstrates:
- Building end-to-end ML pipelines for regression tasks
- Implementing fairness-aware machine learning
- Using gradient boosting for structured data
- Model interpretability for business stakeholders
- Automating HR decision-making processes

## 📝 Future Enhancements

- Incorporate additional features (education, certifications)
- Experiment with ensemble methods
- Real-time prediction API deployment
- Regular model retraining pipeline
- A/B testing framework for production deployment

## 👨‍💻 Author

Machine Learning Capstone Project - T.V.Pavan Kumar

---

**Note**: This model is designed to support HR decision-making and should be used as a tool alongside human judgment to ensure comprehensive evaluation of candidates.
