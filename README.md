# Wisconsin-Breast-Cancer-

📄 Overview
This project leverages the Breast Cancer Wisconsin (Diagnostic) Dataset to develop a machine learning model that predicts whether a breast tumor is malignant or benign. The dataset comprises features computed from digitized images of fine needle aspirate (FNA) of breast masses, describing characteristics of the cell nuclei present.

🎯 Objectives
Data Preprocessing: Clean and prepare the dataset for modeling.
Exploratory Data Analysis (EDA): Understand the underlying patterns and distributions.
Model Development: Train and evaluate machine learning models to accurately classify tumors.
Deployment: Develop a web application to make the model accessible for predictions.

📁 Project Structure
Wisconsin-Breast-Cancer/
├── Data/
│   └── data.csv               # Dataset file
├── app/
│   ├── templates/
│   │   └── index.html         # HTML template for the web app
│   └── app.py                 # Flask application
├── model/
│   └── model.pkl              # Trained machine learning model
├── static/
│   └── styles.css             # CSS styles for the web app
├── .gitignore                 # Git ignore file
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies

🛠️ Setup Instructions
1. Clone the Repository:

git clone https://github.com/0karim0/Wisconsin-Breast-Cancer-.git
cd Wisconsin-Breast-Cancer-

2. Create and Activate a Virtual Environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install Dependencies:

pip install -r requirements.txt

4. Run the Flask Application:

cd app
python app.py

