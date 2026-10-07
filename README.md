# Customer Churn Prediction

An end-to-end machine learning project for predicting whether a telecom customer is likely to churn.

The project is being developed as a structured ML/MLOps workflow, starting from exploratory data analysis and reusable data pipelines and progressing toward model training, experiment tracking, serving, containerization, CI/CD, and cloud deployment.

## Project Objective

Build a reproducible customer churn prediction system using the Telco Customer Churn dataset.

The project focuses on:

- Understanding customer churn patterns
- Building a baseline machine learning model
- Creating reusable data ingestion and transformation components
- Comparing multiple classification models
- Hyperparameter tuning
- Model evaluation and packaging
- API-based model serving
- Containerization
- CI/CD
- Cloud deployment and monitoring

## Dataset

The project uses the Telco Customer Churn dataset.

Dataset characteristics:

- 7,043 customer records
- 21 original columns
- Target variable: `Churn`
- Binary classification problem

The dataset contains customer information such as:

- Demographics
- Tenure
- Contract type
- Internet service
- Payment method
- Monthly charges
- Total charges
- Customer churn status

## Project Structure

```text
customer-churn-prediction/
│
├── data/
│   └── raw/
│       └── Telco-Customer-Churn.csv
│
├── artifacts/
│   ├── raw.csv
│   ├── train.csv
│   ├── test.csv
│   └── preprocessor.pkl
│
├── notebooks/
│   └── customer_churn_analysis.ipynb
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   │
│   ├── exception.py
│   └── logger.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Machine Learning Workflow

```text
Raw Data
   ↓
Data Validation
   ↓
Exploratory Data Analysis
   ↓
Data Preprocessing
   ↓
Feature Engineering
   ↓
Model Training
   ↓
Model Comparison
   ↓
Hyperparameter Tuning
   ↓
Model Packaging
   ↓
Model Serving
   ↓
Containerization
   ↓
CI/CD
   ↓
Cloud Deployment
   ↓
Monitoring & Logging
```

## Course-Aligned Development

The implementation follows the Predictive Analytics course sequence while extending toward a complete ML/MLOps project.

### Unit 3 — Experiment Tracking and Pipeline Structuring
**Completed:**
- Dataset loading
- Data understanding
- Data quality checks
- Exploratory data analysis
- Churn distribution analysis
- Feature relationship analysis
- Baseline Logistic Regression model
- Data story

**Baseline results:**
- Accuracy: 79.91%
- ROC-AUC: 0.8410

### Unit 4 — Data Ingestion and Data Transformation
**Completed:**
- Reusable DataIngestion component
- Train/test split
- Numeric and categorical feature identification
- Numeric preprocessing
- Categorical preprocessing
- ColumnTransformer
- Saved preprocessing artifact

**Preprocessing:**
- Numeric features: Median imputation → StandardScaler
- Categorical features: Most-frequent imputation → OneHotEncoder

**Generated artifacts:**
```text
artifacts/
├── raw.csv
├── train.csv
├── test.csv
└── preprocessor.pkl
```

**Verification:**
- Training rows: 5634
- Testing rows: 1409
- Original features: 19
- Transformed features: 45

### Unit 5 — Model Training and Hyperparameter Tuning
**Current progress:**
- Reused Unit 4 ingestion and transformation components
- Added candidate classification models
- Compared models using 5-fold cross-validation
- Used ROC-AUC as the comparison metric

**Current model comparison:**

| Model | 5-Fold CV ROC-AUC |
| :--- | :--- |
| Logistic Regression | 0.8456 |
| XGBoost | 0.8220 |
| Random Forest | 0.8217 |

**Current best model:**
- Logistic Regression
- Mean CV ROC-AUC: 0.8456

**Next Unit 5 work:**
- Hyperparameter tuning
- Final evaluation on the held-out test set
- Model serialization

## Engineering Roadmap

The project will be extended beyond the course requirements toward an end-to-end ML/MLOps system.

1. **Environment and Library Setup**
   - Python virtual environment and project dependencies.
2. **Exploratory Data Analysis**
   - Understand the dataset, identify data quality issues, analyze churn patterns, and establish a baseline.
3. **Modular ML Pipeline**
   - Move data loading, preprocessing, training, tuning, and evaluation into reusable Python components.
4. **Training, Experiment Tracking, and Hyperparameter Tuning**
   - Planned tools and components include: Scikit-learn, XGBoost, MLflow, Hyperparameter optimization.
5. **Serving Layer**
   - Planned: FastAPI REST API, User-facing prediction interface.
6. **Containerization**
   - Planned: Docker, Reproducible application environment.
7. **CI/CD**
   - Planned: GitHub Actions, Automated testing, Build and deployment workflow.
8. **Cloud Deployment and Monitoring**
   - Planned: AWS, Container registry, Cloud deployment, Application and model monitoring, Logging.

## Tools and Technologies

### Current
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Git
- GitHub

### Planned
- MLflow
- FastAPI
- Docker
- GitHub Actions
- AWS

## Reproducibility

The project is structured so that data processing and model development can be reproduced through reusable Python components rather than relying only on notebook execution.

The preprocessing object is saved as:
`artifacts/preprocessor.pkl`

This allows the same preprocessing logic to be reused when making predictions on new data.

## Logging and Error Handling

The project includes reusable logging and custom exception handling:
- `src/logger.py`
- `src/exception.py`

These components are used by the pipeline modules to provide structured logs and clearer error information.

## Project Status

**Current stage:**
- Unit 3 — Completed
- Unit 4 — Completed
- Unit 5 — Model comparison completed

The next stage is hyperparameter tuning of the strongest candidate model, followed by final held-out test evaluation.

## Author

**Nandith Burla**  
GitHub: [https://github.com/nandithburla](https://github.com/nandithburla)