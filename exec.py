import pandas as pd
import numpy as np

train = pd.read_csv("Train.csv",names=['Item_Identifier', 'Item_Weight', 'Item_Fat_Content','Item_Visibility', 'Item_Type', 'Item_MRP', 'Outlet_Identifier', 'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Item_Outlet_Sales'])
test = pd.read_csv("Test.csv",names=['Item_Identifier', 'Item_Weight', 'Item_Fat_Content','Item_Visibility', 'Item_Type', 'Item_MRP', 'Outlet_Identifier', 'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Item_Outlet_Sales'])
train['source']='train'
test['source']='test'
data = pd.concat([train, test],ignore_index=True)
data.apply(lambda x: sum(x.isnull()))
data.apply(lambda x: len(x.unique()))
categorical_columns = [x for x in data.dtypes.index if data.dtypes[x]=='object']
categorical_columns = [x for x in categorical_columns if x not in ['Item_Identifier','Outlet_Identifier','source']]

#1
item_avg_weight = data.pivot_table(values='Item_Weight', index='Item_Identifier')
data.set_index('Item_Identifier',inplace=True)
data['Item_Weight'].fillna(item_avg_weight.Item_Weight,inplace=True)
# print('Initial #missing: %d'% sum(data['Item_Weight'].isnull()))
data['Item_Weight']=data['Item_Weight'].replace(np.nan,data['Item_Weight'].mean())
data.reset_index(inplace=True)
index = data['Item_Weight'].index[data['Item_Weight'].apply(np.isnan)]
# print(index)
# print('Final #missing: %d'% sum(data['Item_Weight'].isnull()))

# #2
data['Outlet_Size'].fillna(data['Outlet_Size'].mode()[0],inplace=True)
# data['Outlet_Size']=data.groupby(['Item_Identifier']).agg(lambda x:x.value_counts().index[0])
#
# #3
miss_bool = data['Item_Visibility'].isnull()
# # # print('initial : ',sum(miss_bool))
visibility_avg = data.pivot_table(values='Item_Visibility', index='Item_Identifier')
# print(visibility_avg)
data.set_index('Item_Identifier',inplace=True)
data['Item_Visibility'].replace(0,visibility_avg.Item_Visibility,inplace=True)
data.reset_index(inplace=True)
# # print('Number of 0 values after modification: %d'%sum(data['Item_Visibility'] == 0))
# # print(data['Item_Visibility'])
# #
# # #4

data.set_index('Item_Identifier',inplace=True)
data['Item_Visibility_MeanRatio']=np.zeros(len(data.index))
data['Item_Visibility_MeanRatio'].replace(0,data['Item_Visibility']/visibility_avg.Item_Visibility,inplace=True)
data.reset_index(inplace=True)
# data['Item_Visibility_MeanRatio'] = data.apply(lambda x: x['Item_Visibility']/visibility_avg[x['Item_Identifier']], axis=1)
# print(data['Item_Visibility_MeanRatio'].head())

#5
data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])
#Rename them to more intuitive categories:
data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD':'Food','NC':'Non-Consumable','DR':'Drinks'})
data['Item_Type_Combined'].value_counts()

data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']
# print(data['Outlet_Years'].describe())

#6
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF':'Low Fat','reg':'Regular','low fat':'Low Fat'})
# print(data['Item_Fat_Content'].value_counts())

data.loc[data['Item_Type_Combined']=="Non-Consumable",'Item_Fat_Content'] = "Non-Edible"
data['Item_Fat_Content'].value_counts()

#7 One hot encoding

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#New variable for outlet
data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']
le = LabelEncoder()
for i in var_mod:
    data[i] = le.fit_transform(data[i])
data = pd.get_dummies(data, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type','Item_Type_Combined','Outlet'])
# print(data[['Item_Fat_Content_0','Item_Fat_Content_1','Item_Fat_Content_2']].head(10))

#8

data.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)

#Divide into test and train:
train = data.loc[data['source']=="train"]
test = data.loc[data['source']=="test"]

#Drop unnecessary columns:
test.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True)
train.drop(['source'],axis=1,inplace=True)
#Export files as modified versions:
train.to_csv("train_modified.csv",index=False)
test.to_csv("test_modified.csv",index=False)

mean_sales = train['Item_Outlet_Sales'].mean()
#Define a dataframe with IDs for submission:
base1 = test[['Item_Identifier','Outlet_Identifier']]
base1['Item_Outlet_Sales'] = mean_sales
#Export submission file
base1.to_csv("alg0.csv",index=False)

#9

target = 'Item_Outlet_Sales'
IDcol = ['Item_Identifier', 'Outlet_Identifier']
from sklearn import cross_validation, metrics


def modelfit(alg, dtrain, dtest, predictors, target, IDcol, filename):
    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target])

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])

    # Perform cross-validation:
    cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], dtrain[target], cv=20,
                                                scoring='mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))

    # Print model report:
    print("\nModel Report")
    print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(dtrain[target].values, dtrain_predictions)))
    print("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score)))

    # Predict on testing data:
    dtest[target] = alg.predict(dtest[predictors])

    # Export submission file:
    IDcol.append(target)
    submission = pd.DataFrame({x: dtest[x] for x in IDcol})
    submission.to_csv(filename, index=True)

    #10

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet

predictors = [x for x in train.columns if x not in [target] + IDcol]
# print predictors
# alg2 = RandomForestRegressor(n_estimators=400, max_depth=6, min_samples_leaf=100, n_jobs=4)
alg3= ElasticNet(alpha=1, l1_ratio=0.5, normalize=False)
modelfit(alg3, train, test, predictors, target, IDcol, 'alg3.csv')
