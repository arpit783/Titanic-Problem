# -*- coding: utf-8 -*-
"""
Created on Mon May 11 12:39:45 2020

@author: Hp
"""

import pandas as pd
training_dataset = pd.read_csv('train.csv')
import re
pattern_1 = ['Mrs.']
pattern_2 = ['Mr.', 'Rev.', 'Col.', 'Major.', 'Don.', 'Capt.', 'Jonkheer']
pattern_3 = ['Dr.']
pattern_4 = ['Miss.', 'Mlle.', 'Ms.', 'Mme', 'Countess.']
pattern_5 = ['Master.']
for i in range (0, 889):
    for pattern in pattern_1:
        if re.search(pattern, training_dataset['Name'][i]):
            training_dataset['Name'][i] = str('Mrs.')
            training_dataset.ix[i] = training_dataset.ix[i].fillna(35.4)
        else:
            for pattern in pattern_2:
                if re.search(pattern, training_dataset['Name'][i]):
                    training_dataset['Name'][i] = str('Mr.')
                    training_dataset.ix[i] = training_dataset.ix[i].fillna(32.8)
            else:
                for pattern in pattern_3:
                    if re.search(pattern, training_dataset['Name'][i]):
                        training_dataset['Name'][i] = str('Dr.')
                        training_dataset.ix[i] = training_dataset.ix[i].fillna(42)
                else:
                    for pattern in pattern_4:
                        if re.search(pattern, training_dataset['Name'][i]):
                            training_dataset['Name'][i] = str('Miss.')
                            training_dataset.ix[i] = training_dataset.ix[i].fillna(21.9)
                    else:
                        for pattern in pattern_5:
                            if re.search(pattern, training_dataset['Name'][i]):
                                training_dataset['Name'][i] = str('Master.')
                                training_dataset.ix[i] = training_dataset.ix[i].fillna(4.57)
FamilyMembers = []
for j in range (0, 889):
    Total = training_dataset['SibSp'][j] + training_dataset['Parch'][j]
    FamilyMembers.append(Total)
    
training_dataset['FamilyMembers'] = FamilyMembers
    
training_dataset = training_dataset.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Cabin', 'Ticket'], axis = 1)

X_train = training_dataset.iloc[:, 1:7].values
Y_train = training_dataset.iloc[:, 0].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_1 = LabelEncoder()
X_train[:, 1] = label_encoder_1.fit_transform(X_train[:, 1])
label_encoder_2 = LabelEncoder()
X_train[:, 4] = label_encoder_2.fit_transform(X_train[:, 4])

onehotencoder_1 = OneHotEncoder(categorical_features = [4])
X_train = onehotencoder_1.fit_transform(X_train).toarray()
X_train = X_train[:, 1:]
onehotencoder_2 = OneHotEncoder(categorical_features = [3])
X_train = onehotencoder_2.fit_transform(X_train).toarray()
X_train = X_train[:, 1:]

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)

test_dataset = pd.read_csv('test.csv')
import re
pattern_1 = ['Mrs.']
pattern_2 = ['Mr.', 'Rev.', 'Col.', 'Major.', 'Don.', 'Capt.', 'Jonkheer']
pattern_3 = ['Dr.']
pattern_4 = ['Miss.', 'Mlle.', 'Ms.', 'Mme', 'Countess.']
pattern_5 = ['Master.']
for i in range (0, 418):
    for pattern in pattern_1:
        if re.search(pattern, test_dataset['Name'][i]):
            test_dataset['Name'][i] = str('Mrs.')
            test_dataset.ix[i] = test_dataset.ix[i].fillna(35.4)
        else:
            for pattern in pattern_2:
                if re.search(pattern, test_dataset['Name'][i]):
                    test_dataset['Name'][i] = str('Mr.')
                    test_dataset.ix[i] = test_dataset.ix[i].fillna(32.8)
            else:
                for pattern in pattern_3:
                    if re.search(pattern, test_dataset['Name'][i]):
                        test_dataset['Name'][i] = str('Dr.')
                        test_dataset.ix[i] = test_dataset.ix[i].fillna(42)
                else:
                    for pattern in pattern_4:
                        if re.search(pattern, test_dataset['Name'][i]):
                            test_dataset['Name'][i] = str('Miss.')
                            test_dataset.ix[i] = test_dataset.ix[i].fillna(21.9)
                    else:
                        for pattern in pattern_5:
                            if re.search(pattern, test_dataset['Name'][i]):
                                test_dataset['Name'][i] = str('Master.')
                                test_dataset.ix[i] = test_dataset.ix[i].fillna(4.57)
FamilyMembers = []
for j in range (0, 418):
    Total = test_dataset['SibSp'][j] + test_dataset['Parch'][j]
    FamilyMembers.append(Total)
    
test_dataset['FamilyMembers'] = FamilyMembers
    
test_dataset = test_dataset.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Cabin', 'Ticket'], axis = 1)

X_test = test_dataset.iloc[:, 0:6].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_1 = LabelEncoder()
X_test[:, 1] = label_encoder_1.fit_transform(X_test[:, 1])
label_encoder_2 = LabelEncoder()
X_test[:, 4] = label_encoder_2.fit_transform(X_test[:, 4])

onehotencoder_1 = OneHotEncoder(categorical_features = [4])
X_test = onehotencoder_1.fit_transform(X_test).toarray()
X_test = X_test[:, 1:]
onehotencoder_2 = OneHotEncoder(categorical_features = [3])
X_test = onehotencoder_2.fit_transform(X_test).toarray()
X_test = X_test[:, 1:]

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_test = sc_x.fit_transform(X_test)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', decision_function_shape = 'ovo', tol = 1e-2, random_state = 0)
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 10)
accuracies.mean()
accuracies.std()

from sklearn.model_selection import GridSearchCV
parameters = [{'C': [0.1, 1, 100, 1000], 'tol': [1e-2, 1e-3, 1e-4], 'decision_function_shape': ['ovo', 'ovr']}]
grid_search = GridSearchCV(estimator = classifier, cv = 10, n_jobs = -1, scoring = 'accuracy', param_grid = parameters)
grid_search = grid_search.fit(X_train, Y_train)
best_score = grid_search.best_score_
best_score = grid_search.best_params_

