import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_percentage_error


dataset = pd.read_excel("HousePricePrediction.xlsx")

# Printing first 5 records of the dataset
print(dataset.shape)

# data processing
obj = (dataset.dtypes == 'object')
object_cols = list(obj[obj].index)
print("Categorical variables:",len(object_cols))

int_ = (dataset.dtypes == 'integer')
num_cols = list(int_[int_].index)
print("Integer variables:",len(num_cols))

fl = (dataset.dtypes == 'float')
fl_cols = list(fl[fl].index)
print("Float variables:",len(fl_cols))


numerical_dataset = dataset.select_dtypes(include=['number'])

# analysis data
plt.figure(figsize=(12, 6))
sns.heatmap(numerical_dataset.corr(),
            cmap = 'BrBG',
            fmt = '.2f',
            linewidths = 2,
            annot = True)
plt.show()

unique_values = []
for col in object_cols:
  unique_values.append(dataset[col].unique().size)
plt.figure(figsize=(10,6))
plt.title('No. Unique values of Categorical Features')
plt.xticks(rotation=90)
sns.barplot(x=object_cols,y=unique_values)
plt.show()

plt.figure(figsize=(18, 36))
plt.title('Categorical Features: Distribution')
plt.xticks(rotation=90)
index = 1

for col in object_cols:
    y = dataset[col].value_counts()
    plt.subplot(8, 4, index)
    plt.xticks(rotation=90)
    sns.barplot(x=list(y.index), y=y)
    index += 1

plt.show()

# data cleaning
dataset.drop(['Id'],
             axis=1,
             inplace=True)
dataset['SalePrice'] = dataset['SalePrice'].fillna(
  dataset['SalePrice'].mean()) 
new_dataset = dataset.dropna()
new_dataset.isnull().sum()



s = (new_dataset.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables:")
print(object_cols)
print('No. of. categorical features: ', 
      len(object_cols))
OH_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
OH_cols = pd.DataFrame(OH_encoder.fit_transform(new_dataset[object_cols]))
OH_cols.index = new_dataset.index
OH_cols.columns = OH_encoder.get_feature_names_out()

# Concatenate one-hot encoded columns with the rest of the dataset
df_final = new_dataset.drop(object_cols, axis=1)
df_final = pd.concat([df_final, OH_cols], axis=1)

# Define features and target
X = df_final.drop(['SalePrice'], axis=1)
Y = df_final['SalePrice']

# Split the data into training and validation sets
X_train, X_valid, Y_train, Y_valid = train_test_split(
    X, Y, train_size=0.8, test_size=0.2, random_state=0
)

# Train the SVR model
model_SVR = svm.SVR()
model_SVR.fit(X_train, Y_train)

# Predict on the validation set
Y_pred = model_SVR.predict(X_valid)

# Evaluate the model
mape = mean_absolute_percentage_error(Y_valid, Y_pred)
print("Mean Absolute Percentage Error:", mape)

validation_data = pd.read_excel("validation.xlsx")

# Data preprocessing (similar to the training dataset)
validation_data.drop(['Id'], axis=1, inplace=True)  # Drop 'Id' if present
validation_data['SalePrice'] = validation_data['SalePrice'].fillna(validation_data['SalePrice'].mean())
validation_data = validation_data.dropna()

# One-hot encoding for categorical features
validation_data_OH = pd.DataFrame(OH_encoder.transform(validation_data[object_cols]))
validation_data_OH.index = validation_data.index
validation_data_OH.columns = OH_encoder.get_feature_names_out()

# Combine numerical and encoded categorical features
validation_data_final = validation_data.drop(object_cols, axis=1)
validation_data_final = pd.concat([validation_data_final, validation_data_OH], axis=1)

# Drop the target column if it's present in the validation data
if 'SalePrice' in validation_data_final.columns:
    validation_features = validation_data_final.drop(['SalePrice'], axis=1)
else:
    validation_features = validation_data_final

# Predict prices
predicted_prices = model_SVR.predict(validation_features)

# Display the predictions
validation_data['PredictedPrice'] = predicted_prices
print(validation_data[['PredictedPrice']])
