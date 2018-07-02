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

#Set desired print width in InteliJ
desired_width = 500
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)

#Resolve warings
pd.options.mode.chained_assignment = None

'''
#Read input data from csv file
raw_data = pd.read_csv('input/data.csv')

#Create dataset for prediction for final submission
prediction_dataset = raw_data[raw_data['shot_made_flag'].isnull()]

#Create train/test dataset where shot_made_flag not null 
test_data = raw_data[raw_data['shot_made_flag'].notnull()]
test_data.shot_made_flag = test_data['shot_made_flag'].astype(np.int64)

#Clean irelevant columns
test_data = test_data.drop(['game_event_id', 'game_id', 'team_id', 'team_name', 'shot_id', 'game_date', 'matchup'], axis=1)
'''
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

Y = test_data['shot_made_flag']
X = test_data.drop('shot_made_flag', axis=1)

model = LogisticRegression()
rfe = RFE(model, 10)
fit = rfe.fit(X, Y)

print(fit.n_features_)
print(fit.ranking_)
print(fit.support_)
'''

test_data  = pd.read_csv('input/test_data_label_encoded.csv')

#Clean irelevant columns after feature selection (drop: lat, lon, loc_x, loc_y, seconds_remaining, opponent)
test_data = test_data.drop(['lat', 'loc_x', 'loc_y', 'lon', 'seconds_remaining', 'opponent'], axis=1)

#Label Encoding to binary using get_dummies
test_data = pd.get_dummies(test_data,
                           columns=["action_type", "combined_shot_type", "period", "season",
                                    "shot_type", "shot_zone_area", "shot_zone_basic", "shot_zone_range"])

#Split data training/testing
test_data.drop('ggg', axis=1)
#Data with just 'shot_made_flag'
Y = test_data['shot_made_flag']
#Data with features/attributes without 'shot_made_flag'
X_test_data = test_data.drop('shot_made_flag', axis=1)

#Splitting data
X_train, X_test, y_train, y_test = train_test_split(X_test_data, Y, test_size=0.2)

#Testing different classifiers

# Linear Regression
#model = LinearRegression().fit(X_train, y_train)
# Logistic Regression
#model = LogisticRegression().fit(X_train, y_train)
# Decision Tree
#model = DecisionTreeClassifier(min_samples_split=3000, random_state=99).fit(X_train, y_train)
# Random Forest
model = RandomForestClassifier(n_estimators=100, max_features='auto', warm_start=True, max_depth=10).fit(X_train, y_train)
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


#rfe = RFE(LogisticRegression(), 17)
#rfe.fit(X_train, y_train)

print ("Score:" + str(model.score(X_test, y_test)))

# Perform 6-fold cross validation
scores = cross_val_score(model, X_test_data, Y, cv=10)
print ("Cross validated score:")
print(scores)
print('Mean: ' + str(np.mean(scores)))

#print(len(train))
#print(len(test))

#print('Shape train' + str(train.shape))
#print('Shape test' + str(test.shape))

#print('TEST 1 2 3 4 5 6 7 8 9 10')
#print(test_data.dtypes)


# prepare cross validation
#kfold = KFold(10)

# enumerate splits
#for train, eval in kfold.split(test_data):
#    print('train indexes: %s, eval indexes: %s' % (test_data.iloc[train,:], test_data.iloc[eval, :]))

#print('Column count after cleanup: ' + str(len(test_data.columns)))
#print('Row count after cleanup: ' + str(len(test_data.index)))

#print('Test data prepared:')
#print(test_data)

#prediction_dataset.to_csv('input/prediction_dataset.csv')
#test_data.to_csv('input/test_data.csv')

#input.to_csv('output/output.csv')
#plt.interactive(False)
#print raw_input.plot(figsize=(30, 30))
#plt.show()