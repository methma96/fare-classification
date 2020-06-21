from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import train_test_split
from pyproj import Geod 
from numpy import mean
import seaborn as sns;sns.set()
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
from imblearn.pipeline import Pipeline
from catboost import CatBoostClassifier
RANDOM_SEED = 8    # Set a random seed for reproducibility!

import numpy as np
import pandas as pd

df = pd.read_csv("/home/nethma/ML/train.csv")
X = df.drop(columns=["tripid", "label"])
y = df["label"].map({'incorrect': 0, 'correct': 1})

X['drop_lon'].replace({45.3077: np.nan}, inplace = True) #replace longtitudes located outside the country.
X['drop_lat'].replace({48.132: np.nan}, inplace = True)#replace latitude located outside the country

wgs84_geod = Geod(ellps='WGS84') #Distance will be measured on this ellipsoid - more accurate than a spherical method

#Get distance between pairs of lat-lon points
def Distance(lat1,lon1,lat2,lon2):
  az12,az21,dist = wgs84_geod.inv(lon1,lat1,lon2,lat2) 
  return dist

X["pickup_time"] = pd.to_datetime(X["pickup_time"],errors = "coerce")
X["drop_time"] = pd.to_datetime(X["drop_time"],errors = "coerce")
X["pickup_time_hour"] = X["pickup_time"].dt.hour
X["pickup_time_minute"] = X["pickup_time"].dt.minute
X["drop_time_hour"] =X["drop_time"].dt.hour
X["drop_time_minute"] =X["drop_time"].dt.minute
X["pickup_time_day"] = X["pickup_time"].dt.day
X["drop_time_day"] = X["drop_time"].dt.day
X["pickup_time"] = pd.to_numeric(X["pickup_time"] )
X["drop_time"] = pd.to_numeric(X["drop_time"])
X['distance'] = Distance(X['pick_lat'].tolist(),X['pick_lon'].tolist(),
                                   X['drop_lat'].tolist(),X['drop_lon'].tolist())
X["effective_time"] = X["duration"]-X["meter_waiting"]                                 
# chain preprocessing into a Pipeline object
# each step is a tuple of (name you chose, sklearn transformer)
model = CatBoostClassifier(iterations=500000)
numeric_cols = X.columns[X.dtypes != "object"].values
non_numeric_cols = X.columns[X.dtypes == 'object'].values

# chain preprocessing into a Pipeline object
# each step is a tuple of (name you chose, sklearn transformer)
numeric_preprocessing_steps = Pipeline(steps=[
    	('imputer', SimpleImputer(strategy='median')),
    	('scaler', StandardScaler())])
non_numeric_preprocessing_steps = Pipeline(steps=[
	('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
	('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers = [("numeric", numeric_preprocessing_steps, numeric_cols),("non_numeric",non_numeric_preprocessing_steps,non_numeric_cols)],remainder = "drop")

full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model),
])

X_train, X_eval, y_train, y_eval = train_test_split(X,y,test_size=0.33,shuffle=True,stratify=y,random_state=RANDOM_SEED)


full_pipeline.fit(X_train, y_train)

preds = full_pipeline.predict(X_eval)

#evaluate the model
print(f1_score(y_eval, preds, average='macro'))



# # Train model
full_pipeline.fit(X, y)

sns.heatmap(X.corr())
plt.show()
test_features_df = pd.read_csv("/home/nethma/ML/test.csv", 
                               index_col="tripid")
test_features_df["pickup_time"] = pd.to_datetime(test_features_df["pickup_time"],errors = "coerce")
test_features_df["drop_time"] = pd.to_datetime(test_features_df["drop_time"],errors = "coerce")
test_features_df["pickup_time_hour"] = test_features_df["pickup_time"].dt.hour
test_features_df["pickup_time_minute"] = test_features_df["pickup_time"].dt.minute
test_features_df["drop_time_hour"] =test_features_df["drop_time"].dt.hour
test_features_df["drop_time_minute"] =test_features_df["drop_time"].dt.minute
test_features_df["effective_time"] = test_features_df["duration"]-test_features_df["meter_waiting"]
test_features_df["pickup_time_day"] = test_features_df["pickup_time"].dt.day
test_features_df["drop_time_day"] = test_features_df["drop_time"].dt.day

test_features_df['distance'] = Distance(test_features_df['pick_lat'].tolist(),test_features_df['pick_lon'].tolist(),
                                   test_features_df['drop_lat'].tolist(),test_features_df['drop_lon'].tolist())
test_features_df["pickup_time"] = pd.to_numeric(test_features_df["pickup_time"] )
test_features_df["drop_time"] = pd.to_numeric(test_features_df["drop_time"])
test_features_df['distance'] = Distance(test_features_df['pick_lat'].tolist(),test_features_df['pick_lon'].tolist(),
                                   test_features_df['drop_lat'].tolist(),test_features_df['drop_lon'].tolist())
      
test_probas = full_pipeline.predict(test_features_df)
	
# Save predictions to submission data frame
submission_df = pd.read_csv("/home/nethma/ML/sample_submission.csv", 
                            index_col="tripid")
submission_df.head()

# Make sure we have the rows in the same order
np.testing.assert_array_equal(test_features_df.index.values, 
                              submission_df.index.values)

submission_df["prediction"] = test_probas


submission_df.head()

submission_df.to_csv('my_submission2.csv', index=True)






