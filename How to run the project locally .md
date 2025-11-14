 **HOW TO RUN THE STUDENT DROPOUT PREDICTION PROJECT LOCALLY ON YOUR COMPUTER**
1. Download or Clone the Project

You can obtain the my full project in either of two ways:

Download ZIP from GitHub and extract the folder to your computer.

Or clone using Git:

git clone https://github.com/Hlabanyane/Student-Dropout-Model.git

2. Install Python

Install Python from:

 https://www.python.org/downloads/

During installation, ensure you check:

 Add Python to PATH

3. Open the Project Folder

Navigate into the project directory using Terminal or PowerShell:

cd Student-Dropout-Model

4. Create and Activate a Virtual Environment
Windows
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
venv\Scripts\activate

Mac / Linux
python3 -m venv venv
source venv/bin/activate


After activation, your terminal should show:

(venv)

5. Install Dependencies

Install all required packages:

pip install -r requirements.txt


This installs:

pandas

numpy

joblib

matplotlib

seaborn

scikit-learn

streamlit
…and more.

6. Add the Dataset

Ensure your data file is placed in the project directory:

 student_data.csv

This file is required for training the model.

7. Train the Model (Optional)

If you want to retrain the model using your own dataset:

python main_model.py


Running this script will create:

student_model.pkl

scaler.pkl

label_encoders.pkl

feature_names.pkl

These files are required for prediction in the dashboard.

8. Run the Streamlit Dashboard

Start the web application using:

streamlit run app.py


The dashboard will automatically open in your browser at:

 http://localhost:8501

9. How to Use the Dashboard
Single Prediction

Enter student details manually.

Click Predict Outcome

Download the prediction as CSV.

Batch Prediction

Upload a CSV or Excel file containing multiple student records.

The system processes all data and predicts outcomes.

Download the full prediction dataset as a CSV.

Visualizations

The dashboard displays:

Boxplot showing missing values

Pie chart showing distribution of student outcomes

Clean table of input data

Exportable predictions

10. Exit the Virtual Environment

When you are finished:

deactivate
