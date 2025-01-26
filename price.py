import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm

# Load dataset
dataset = pd.read_excel("HousePricePrediction.xlsx")

# Data cleaning and preprocessing
dataset.drop(['Id'], axis=1, inplace=True)
dataset['SalePrice'] = dataset['SalePrice'].fillna(dataset['SalePrice'].mean())
new_dataset = dataset.dropna()

# Identify categorical and numerical features
object_cols = new_dataset.select_dtypes(include='object').columns.tolist()

OH_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
OH_cols = pd.DataFrame(OH_encoder.fit_transform(new_dataset[object_cols]))
OH_cols.index = new_dataset.index
OH_cols.columns = OH_encoder.get_feature_names_out()

# Combine processed data
df_final = pd.concat([new_dataset.drop(object_cols, axis=1), OH_cols], axis=1)

# Define features and target
X = df_final.drop(['SalePrice'], axis=1)
Y = df_final['SalePrice']

# Split data
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=0)

# Train SVR model
model_SVR = svm.SVR()
model_SVR.fit(X_train, Y_train)

# Function to get user input for categorical columns
def get_categorical_input(col, unique_values):
    while True:
        user_input = input(f"Enter {col} ({', '.join(unique_values)}): ")
        if user_input in unique_values:
            return user_input
        else:
            print(f"Invalid input! Please enter one of the following: {', '.join(unique_values)}")

# Simplified predict function
def predict_sale_price():
    # User input fields
    user_input = {}

    # Get user inputs for categorical features
    categorical_inputs = {
        'MSZoning': ['RL', 'RM', 'C', 'FV', 'RH'],
        'LotConfig': ['Inside', 'FR2', 'Corner', 'CulDSac', 'FR3'],
        'BldgType': ['1Fam', '2fmCon', 'Duplex', 'TwnhsE', 'Twnhs'],
        'Exterior1st': ['VinylSd', 'MetalSd', 'Wd Sdng', 'HdBoard', 'BrkFace', 'WdShing', 'CemntBd', 'Plywood', 'AsbShng', 'Stucco', 'BrkComm', 'AsphShn', 'Stone', 'ImStucc', 'CBlock', 'nan']
    }

    # Ask for user inputs for each categorical variable
    for col, values in categorical_inputs.items():
        user_input[col] = get_categorical_input(col, values)

    # Ask for numerical inputs
    user_input['MSSubClass'] = float(input("Enter MSSubClass: "))
    user_input['LotArea'] = float(input("Enter LotArea: "))
    user_input['OverallCond'] = float(input("Enter OverallCond: "))
    user_input['YearBuilt'] = float(input("Enter YearBuilt: "))
    user_input['YearRemodAdd'] = float(input("Enter YearRemodAdd: "))
    user_input['BsmtFinSF2'] = float(input("Enter BsmtFinSF2: "))
    user_input['TotalBsmtSF'] = float(input("Enter TotalBsmtSF: "))

    # Create a DataFrame for the user input
    input_data = pd.DataFrame([user_input])

    # One-hot encode the categorical features in user input
    input_encoded = pd.DataFrame(OH_encoder.transform(input_data[categorical_inputs.keys()]))
    input_encoded.columns = OH_encoder.get_feature_names_out()
    input_encoded.index = input_data.index

    # Combine numerical and encoded categorical data
    input_data = pd.concat([input_data.drop(categorical_inputs.keys(), axis=1), input_encoded], axis=1)

    # Align input data with training data columns
    input_data = input_data.reindex(columns=X_train.columns, fill_value=0)

    # Predict and display the sale price
    predicted_price = model_SVR.predict(input_data)
    print(f"Predicted Sale Price: {predicted_price[0]}")

# Run prediction
predict_sale_price()
