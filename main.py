import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.metrics import accuracy_score

#Feature manipulation
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.cross_validation import cross_val_score, cross_val_predict

#Classifiers
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier

#InteliJ configure desired print width
desired_width = 500
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)

#Resolve warings
pd.options.mode.chained_assignment = None

#Read input data from csv file
raw_data = pd.read_csv('input/data.csv')

#Clean irelevant columns
raw_data = raw_data.drop(['game_event_id', 'game_id', 'team_id', 'team_name', 'game_date', 'matchup'], axis=1)

#Label Encoding to binary using get_dummies
raw_data = pd.get_dummies(raw_data,
                           columns=["action_type", "combined_shot_type", "period", "season",
                                    "shot_type", "shot_zone_area", "shot_zone_basic", "shot_zone_range", "opponent"])

#Create dataset for prediction for final submission
test_data = raw_data[raw_data['shot_made_flag'].isnull()]

#Create train/test dataset where shot_made_flag not null 
training_data = raw_data[raw_data['shot_made_flag'].notnull()]
#test_data.shot_made_flag = test_data['shot_made_flag'].astype(np.int64)
training_data = training_data.drop(['shot_id'], axis=1)

'''
#Feature selection (choosing which features can be dropped)
#Temporary encode categorical data to integer for feature selection
test_data.action_type=test_data.loc[:,'action_type'].apply(lambda x :  np.where(x == test_data.action_type.unique())[0][0])
test_data.combined_shot_type=test_data.loc[:,'combined_shot_type'].apply(lambda x :  np.where(x == test_data.combined_shot_type.unique())[0][0])
test_data.lat=test_data.loc[:,'lat'].apply(lambda x :  np.where(x == test_data.lat.unique())[0][0])
test_data.loc_x=test_data.loc[:,'loc_x'].apply(lambda x :  np.where(x == test_data.loc_x.unique())[0][0])
test_data.loc_y=test_data.loc[:,'loc_y'].apply(lambda x :  np.where(x == test_data.loc_y.unique())[0][0])
test_data.lon=test_data.loc[:,'lon'].apply(lambda x :  np.where(x == test_data.lon.unique())[0][0])
test_data.minutes_remaining=test_data.loc[:,'minutes_remaining'].apply(lambda x :  np.where(x == test_data.minutes_remaining.unique())[0][0])
test_data.period=test_data.loc[:,'period'].apply(lambda x :  np.where(x == test_data.period.unique())[0][0])
test_data.playoffs=test_data.loc[:,'playoffs'].apply(lambda x :  np.where(x == test_data.playoffs.unique())[0][0])
test_data.season=test_data.loc[:,'season'].apply(lambda x :  np.where(x == test_data.season.unique())[0][0])
test_data.seconds_remaining=test_data.loc[:,'seconds_remaining'].apply(lambda x :  np.where(x == test_data.seconds_remaining.unique())[0][0])
test_data.shot_distance=test_data.loc[:,'shot_distance'].apply(lambda x :  np.where(x == test_data.shot_distance.unique())[0][0])
test_data.shot_type=test_data.loc[:,'shot_type'].apply(lambda x :  np.where(x == test_data.shot_type.unique())[0][0])
test_data.shot_zone_area=test_data.loc[:,'shot_zone_area'].apply(lambda x :  np.where(x == test_data.shot_zone_area.unique())[0][0])
test_data.shot_zone_basic=test_data.loc[:,'shot_zone_basic'].apply(lambda x :  np.where(x == test_data.shot_zone_basic.unique())[0][0])
test_data.shot_zone_range=test_data.loc[:,'shot_zone_range'].apply(lambda x :  np.where(x == test_data.shot_zone_range.unique())[0][0])
test_data.opponent=test_data.loc[:,'opponent'].apply(lambda x :  np.where(x == test_data.opponent.unique())[0][0])

#test_data  = pd.read_csv('input/test_data_label_encoded.csv')

Y = test_data['shot_made_flag']
X = test_data.drop('shot_made_flag', axis=1)

model = RandomForestClassifier()
rfe_model = RFE(model, 10, step=1)
rfe_model = rfe_model.fit(X, Y)

print('Num features: ' + str(rfe_model.n_features_))
print('Feature rainkins: ' + str(rfe_model.ranking_))
print('Feature support: ' + str(rfe_model.support_))

#test_data  = pd.read_csv('input/test_data_label_encoded.csv')

#Clean irelevant columns after feature selection (drop: lat, lon, loc_x, loc_y, seconds_remaining, opponent)
test_data = test_data.drop(['lat', 'loc_x', 'loc_y', 'lon', 'seconds_remaining', 'opponent'], axis=1)
'''

#Split data training/testing
#test_data.drop('ggg', axis=1)z
#Data with just 'shot_made_flag'
Y_train_data = training_data['shot_made_flag']
#Data with features/attributes without 'shot_made_flag'
X_train_data = training_data.drop('shot_made_flag', axis=1)

'''
#Splitting data
X_train, X_test, y_train, y_test = train_test_split(X_train_data, Y_train_data, test_size=0.25)

#Testing different classifiers
# Linear Regression
#model = LinearRegression().fit(X_train, y_train)
# Logistic Regression
#model = LogisticRegression().fit(X_train, y_train)
# Decision Tree
#model = DecisionTreeClassifier(min_samples_split=3000, random_state=99).fit(X_train, y_train)
# Random Forest
model = RandomForestClassifier(n_estimators=100, max_features='auto', warm_start=True, max_depth=10)
# Extreemly Randomized Trees
#model = ExtraTreesClassifier(n_estimators=200, max_features=30).fit(X_train, y_train)
# Multi Layer Perceptron
#model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20,), random_state=1).fit(X_train, y_train)
# Gaussian Naive Bayes
#model = GaussianNB().fit(X_train, y_train)
# Stochastic Gradient Descent
#model = SGDClassifier(loss="hinge", penalty="l2").fit(X_train, y_train)
# Support Vector Regression
#model = svm.SVR().fit(X_train, y_train)
# XGBoost
#model = XGBClassifier().fit(X_train, y_train).fit(X_train, y_train)

model = RFE(model, 80, step=1)
model = model.fit(X_train, y_train)

print('Toral features: ' + str(len(X_train_data.columns)) + ' Num features after selection: ' + str(model.n_features_))
print ('Model Score: ' + str(model.score(X_test, y_test)))

#print('Count of all params: ' + str(len(X.columns)))
#print('fit.get_params() count: ' + str(len(fit.get_params())))
#print('Print parameters: ' + str(fit.get_params()))

# Perform 10-fold cross validation
scores = cross_val_score(model, X_test, y_test, cv=10)
print ("Cross validation individual scores: " + str(scores))
print('Cross validation mean score: ' + str(np.mean(scores)))
'''
#Create final model submission csv

print('Training final model on all data')
final_model = RandomForestClassifier(n_estimators=100, max_features='auto', warm_start=True, max_depth=10)
final_model = RFE(final_model, 80, step=1)
final_model = final_model.fit(X_train_data, Y_train_data)

ID=test_data["shot_id"]
test_data = test_data.drop(["shot_id","shot_made_flag"], axis=1)

y_test_data = final_model.predict(test_data)
print('y_test_data' + str(y_test_data))

final_submission=pd.DataFrame({"shot_id":ID,"shot_made_flag":y_test_data})
final_submission.to_csv('output/submission.csv', index=False)

print('Program execution completed')