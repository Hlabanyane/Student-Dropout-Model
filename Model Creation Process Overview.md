Model Creation Process Overview

1. Data Gathering and Cleaning

The dataset was loaded using Pandas and cleaned by removing irrelevant features such as parental education and admission order.

Missing values in numerical columns were replaced with the mean, while categorical variables were filled using the mode.

Outliers were detected and handled using the Interquartile Range (IQR) method to cap extreme values.

Descriptive statistical analysis (mean, median, standard deviation, and distribution plots) was performed to understand the data’s characteristics.

2. Data Preprocessing

Label Encoding was applied to categorical variables to convert text into numeric values.

StandardScaler was used to normalize numeric data, ensuring all features contributed equally to model learning.

The dataset was then divided into training (80%) and testing (20%) subsets using the train_test_split() method.

3. Model Selection

The Random Forest Classifier was chosen for its high accuracy, resistance to overfitting, and ability to handle complex, nonlinear relationships.

This model combines multiple decision trees to improve prediction stability and generalization.

4. Model Training

The Random Forest model was trained using the training data to learn patterns that differentiate between dropouts, enrolled students, and graduates.

The model’s performance was initially assessed using a baseline accuracy score of 77%.

5. Model Tuning

GridSearchCV was used for hyperparameter tuning.

The best parameters found were n_estimators=100, max_depth=None, and min_samples_split=5.

The tuned model improved accuracy to 78%.

6. Model Evaluation

Evaluation was done using metrics such as accuracy, precision, recall, F1-score, and the confusion matrix.

The confusion matrix showed that most graduates were correctly classified, while dropout prediction improved slightly after tuning.

7. Model Deployment

The trained and tuned model, along with the scaler and encoders, was saved using Joblib for integration with the Streamlit dashboard.

The dashboard supports single and batch predictions and allows users to download prediction results in CSV format.
