import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn import metrics
df = pd.read_csv('Car Data.csv')
df.head()
df.shape
print(df['Seller_Type'].unique())
print(df['Transmission'].unique())
print(df['Owner'].unique())
print(df['Fuel_Type'].unique())
df.isnull().sum()
df.describe()
df.columns
final_dataset = df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven','Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]
final_dataset.head()
final_dataset['Current_year']=2022
final_dataset.head()
final_dataset['No_of_year']=final_dataset['Current_year']-final_dataset['Year']
final_dataset.head()
final_dataset.drop(['Current_year'],axis=1,inplace=True)
final_dataset.head()
final_dataset=pd.get_dummies(final_dataset,drop_first=True)
final_dataset.head()
final_dataset.corr()
sns.pairplot(final_dataset)
cormat=final_dataset.corr()
top_corr_features=cormat.index
plt.figure(figsize=(20,20))
#plot heap map
g=sns.heatmap(final_dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")
final_dataset.head()
#independent and dependent feature
x=final_dataset.iloc[:,2:]
y=final_dataset.iloc[:,1]
x.head()
y.head()
#feature importance
from sklearn.ensemble import ExtraTreesRegressor
model =ExtraTreesRegressor()
model.fit(x,y)
print(model.feature_importances_)
#plot graph of features importance for better visulization
feat_importances= pd.Series(model.feature_importances_,index=x.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
x_train.shape
from sklearn.ensemble import RandomForestRegressor
rf_random = RandomForestRegressor()
#hyperparameter
import numpy as np
n_estimators=[int(x) for x in np.linspace(start=100, stop=1200,num=12)]
print(n_estimators)
from sklearn.model_selection import RandomizedSearchCV
 #Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)

rf_random.fit(x_train,y_train)

predictions=rf_random.predict(x_test)
predictions
sns.displot(y_test-predictions)
plt.scatter(y_test,predictions)
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
import pickle
# open a file, where you ant to store the data
file = open('random_forest_regression_model.pkl', 'wb')

# dump information to that file
pickle.dump(rf_random,file)