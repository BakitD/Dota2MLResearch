import pandas
import datetime

import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

NULL_FILLER = 0
TARGET_VALUE = 'radiant_win'
TREE_TEST_NUMBER = [10, 20 ,30, 40, 50, 60]
REGR_C_START = 1e-5
REGR_C_END = 60
REGR_C_STEP = 10
FETURES_TO_REMOVE = ['duration',
                    'tower_status_radiant',
					'tower_status_dire',
					'barracks_status_radiant',
					'barracks_status_dire']
CATEG_TO_REMOVE = ['lobby_type', 'r1_hero', 'r2_hero', 'r3_hero','r4_hero', 'r5_hero',
					'd1_hero', 'd2_hero', 'd3_hero','d4_hero', 'd5_hero']


UV_FEATURES = ['r1_hero', 'r2_hero', 'r3_hero','r4_hero', 'r5_hero',
			   'd1_hero', 'd2_hero', 'd3_hero','d4_hero', 'd5_hero']


features = pandas.read_csv('features.csv', index_col='match_id')

''' Removing unnecessary features '''
def removeFeatures(data, categories):
	for field in categories:
		data = data.drop(field, 1)
	return data

features = removeFeatures(features, FETURES_TO_REMOVE)


''' Looking for columns with nulls '''
def findEmptyColumns(data):
	empty_columns = []
	total = 0
	for column in data.columns.values:
		difference = abs(data[column].isnull().count() - data[column].count())
		if difference:
			total += difference
			empty_columns.append({'name': column, 'difference': difference})
		else:
			total += data[column].count()
	return (empty_columns, total)


def emptyOutput(data, completed):
	total = 0
	for element in data:
		omissions = element['difference']
		name = element['name']
		total += omissions
		print('column name {}, omissions: {}'.format(name, omissions))
	percent = completed / (completed + total) * 100;
	print('Total omissions {}, complete {}, percent of completed {:.2f}'.format(total, completed, percent))

'''
According to results data is completed
For feature firts_blood_player1
value NULL when 5 minutes was passed after game started
but there was not blood. Therefore features first_blood_time
and first_blood_team are equal and has same value as first_blood_player1.
For feature radiant_flying_courier_time NULL means that no one
got flying_courier during game process.
'''
emptyOutput(*findEmptyColumns(features))

def fillNullValues(data, filler = NULL_FILLER):
	return data.fillna(filler)

''' Filling null values '''
features = fillNullValues(features)


''' Target variable is radiant_win '''
def divideData(data):
	return data.drop(TARGET_VALUE, 1), data[TARGET_VALUE]

xTrain, yTrain = divideData(features)


def treeBoost(n_estimators, x, y, kfold):
    boosting = GradientBoostingClassifier(n_estimators=n_estimators, random_state=241)
    return np.mean(cross_val_score(boosting, x, y, cv=kfold, scoring='roc_auc', n_jobs=-1))

''' Gradient boosting '''

def evaluateBoosting(x, y):
    results = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=241)
    for tree_number in TREE_TEST_NUMBER:
        start = datetime.datetime.now()
        result = treeBoost(tree_number, x, y, kfold)
        end = datetime.datetime.now() - start
        results.append({'roc': result, 'time': end, 'trees': tree_number})
    return results

def resultOutput(results, parameter, roc='roc', time='time'):
    for result in results:
        print('ROC {:.5f} time {} {} {}'.format(result.get(roc),
                                                result.get(str(time)),
                                                parameter,
                                                result.get(parameter)))

resultOutput(evaluateBoosting(xTrain, yTrain), 'trees')


def logisticRegression(c, x, y, kfold):
	logistic = LogisticRegression(C=c, random_state=241)
	return np.mean(cross_val_score(logistic, x, y, cv=kfold, scoring='roc_auc', n_jobs=-1))


def evaluateRegression(x, y):
	results = []
	kfold = KFold(n_splits=5, shuffle=True, random_state=241)
	c = REGR_C_START
	while (c <= REGR_C_END):
		start = datetime.datetime.now()
		result = logisticRegression(c, x, y, kfold)
		end = datetime.datetime.now() - start
		results.append({'roc': result, 'time': end, 'c': c})
		c *= REGR_C_STEP
	return results


scaledX = StandardScaler().fit_transform(xTrain)
resultOutput(evaluateRegression(scaledX, yTrain), 'c')


''' Getting unique feature values '''
def getUniqueHeroes(data, features):
	unique_values = np.unique(data[features].fillna(0))
	return unique_values, len(unique_values)

print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Counting unique values')
uHeroes, uCount = getUniqueHeroes(features, UV_FEATURES)
print('Count {}, values {}'.format(uCount, uHeroes))

'''Removing categorized features '''
xTrainNoHeroes = removeFeatures(xTrain, CATEG_TO_REMOVE)
scaledX = StandardScaler().fit_transform(xTrainNoHeroes)
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Removing categorized')
resultOutput(evaluateRegression(scaledX, yTrain), 'c')

''' Bag of words '''
def changeHeroes(data, unique_values, N):
    X_pick = np.zeros((data.shape[0], N))
    for i, match_id in enumerate(data.index):
        for p in range(1,6):
            X_pick[i, np.where(unique_values == data.ix[match_id, 'r%d_hero' % p])] = 1
            X_pick[i, np.where(unique_values == data.ix[match_id, 'd%d_hero' % p])] = -1
    return X_pick


bagX = changeHeroes(xTrain, uHeroes, uCount)
bagXTrain = xTrainNoHeroes.join(pandas.DataFrame(bagX, index=features.index))
scaledX = StandardScaler().fit_transform(bagXTrain)
# Step 5
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Step 5')
resultOutput(evaluateRegression(scaledX, yTrain), 'c')
# scaledX == XX4

# step 6
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Step 6')
featuresTest = pandas.read_csv('features_test.csv', index_col='match_id')
xTest = featuresTest
xTest = xTest.fillna(0)
x2Test = removeFeatures(xTest, CATEG_TO_REMOVE)
x4Test = x2Test.join(pandas.DataFrame(changeHeroes(xTest, uHeroes, uCount),
											  index=featuresTest.index))

xTestFinal = StandardScaler().fit_transform(bagXTrain)
#xTestFinal = StandardScaler().fit_transform(x4Test)

logReg = LogisticRegression(C=1e-5)
logReg.fit(xTestFinal, yTrain)

xx4Test = StandardScaler().fit_transform(x4Test)
result = logReg.predict_proba(xx4Test)
print(result.min(), result.max())
print(result)