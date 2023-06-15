import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder, StandardScaler, power_transform
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle

# Load your dataset
data = pd.read_csv('Clean_data.csv')

# Separate features and target variable
X = data.drop(['Car_price','Unnamed: 0'], axis=1)
y = data['Car_price']

class skewness_remove:
  def __init__(self,skew=0.5):
    self.skew=skew
  
  def fit(self,X,y=None):
    return self

  def transform(self,X):
    x=X.copy()
    X_num=X.select_dtypes(exclude='object')
    skewness=X_num.apply(lambda x:x.skew())
    skewness_col=skewness[abs(skewness)>=self.skew].index
    # Convert the data type to numeric
    X[skewness_col] = X[skewness_col].apply(pd.to_numeric, errors='coerce')
    X[skewness_col]=power_transform(X[skewness_col])
    return X

sk = skewness_remove()
sk.fit(X)

class Encoding:
  def __init__(self):
    pass
  def fit(self,X,y=None):
    return self
  def transform(self,X):
    le=LabelEncoder()
    cols= ['color', 'front_brake_type', 'rear_brake_type', 'Ownership', 'Steering_Type',
       'Brand', 'Model', 'Fuel_type', 'Gear_Transmission']
    for c in cols:
      X[c]=le.fit_transform(X[c])
      X[c]
    return X

enc = Encoding()
enc.fit(X)
enc.transform(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

# Create an instance of the StandardScaler
scaler = StandardScaler()
# Fit the scaler to x_train and transform both x_train and x_test
scaler_ = scaler.fit(X_train)

import joblib

model = joblib.load('UsedCar_price_prediction.pkl')


# Streamlit app code
st.title('Used Car Price Estimation/Prediction')
st.write('Please enter the input features for prediction:')

# Create input fields for each feature

categorical_features = ['color', 'front_brake_type', 'rear_brake_type', 'Ownership', 'Steering_Type',
       'Brand', 'Model', 'Fuel_type', 'Gear_Transmission']
input_features = []
for feature in X.columns:
    if feature in categorical_features:  # List of categorical feature names
        input_feature = st.selectbox(f'Enter {feature}', data[feature].unique())
    else:
        input_feature = st.number_input(f'Enter {feature}', value=0.0)
    input_features.append(input_feature)

# Convert input features to a DataFrame
input_df = pd.DataFrame([input_features], columns=X.columns)

# Preprocess the input data
#preprocessed_input = sk.transform(input_df)
preprocessed_input = enc.transform(input_df)
# Scale numerical features using StandardScaler
preprocessed_input = scaler_.transform(preprocessed_input)


# Make predictions
prediction = model.predict(preprocessed_input)

st.write('Predicted Car Price:', prediction[0])
