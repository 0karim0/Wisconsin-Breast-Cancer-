import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

def create_model(data): 
  X = data.drop(['diagnosis'], axis=1)
  y = data['diagnosis']
  
  # scale the data
  scaler = StandardScaler()
  X = scaler.fit_transform(X)
  
  # split the data
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
  )
  
  # train the model
  model = LogisticRegression()
  model.fit(X_train, y_train)
  
  # test model
  y_pred = model.predict(X_test)
  print('Accuracy of our model: ', accuracy_score(y_test, y_pred))
  print("Classification report: \n", classification_report(y_test, y_pred))
  
  return model, scaler


def get_clean_data():
  data = pd.read_csv("Wisconsin-Breast-Cancer-/Data/data.csv")
  
  data = data.drop(['Unnamed: 32', 'id'], axis=1)
  
  data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })
  
  return data




def save_model_and_scaler(model, scaler):
    # Specify the existing 'model' directory
    model_dir = 'Wisconsin-Breast-Cancer-/model'  
    
    # Ensure that the directory exists
    if not os.path.exists(model_dir):
        print(f"Error: Directory '{model_dir}' does not exist.")
        return
    
    # Save the model to 'model/model.pkl'
    model_path = os.path.join(model_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved successfully to {model_path}")

    # Save the scaler to 'model/scaler.pkl'
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved successfully to {scaler_path}")


def main():
  data = get_clean_data()

  model, scaler = create_model(data)

  save_model_and_scaler(model, scaler)

if __name__ == '__main__':
  main()