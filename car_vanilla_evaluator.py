# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 12:56:10 2017

@author: Hugot
"""
import numpy as np

# Part 1 : load data

Xt = np.genfromtxt('car.data', delimiter=',', dtype=[
        'U15','U15','U15', 'U15','U15','U15','U15'], names=(
        'buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety',
        'class'))

#dict_buying = {'low':1, 'med':2, 'high':3, 'vhigh':4}
#dict_maint  = {'low':1, 'med':2, 'high':3, 'vhigh':4}
#dict_doors = {'2':2, '3':3, '4':4, '5more':5}
#dict_persons = {'2':2, '4':4, 'more':5}
#dict_lug_boot = {'small':1, 'med':2, 'big':3}
#dict_safety = {'low':1, 'med':2, 'high':3}
#dict_class = {'unacc':0, 'acc':1, 'good':2, 'vgood':3}

# ordered encoding : (i - 0.5)/N
dict_buying = {'low':0.125, 'med':0.375, 'high':0.625, 'vhigh':0.875}
dict_maint  = {'low':0.125, 'med':0.375, 'high':0.625, 'vhigh':0.875}
dict_doors = {'2':0.125, '3':0.375, '4':0.625, '5more':0.875}
dict_persons = {'2':0.167, '4':0.500, 'more':0.833}
dict_lug_boot = {'small':0.167, 'med':0.500, 'big':0.833}
dict_safety = {'low':0.167, 'med':0.500, 'high':0.833}
dict_class = {'unacc':0, 'acc':1, 'good':2, 'vgood':3}

#Xtt=Xt[0:3] # row 0,1,2
#Xtt_num = np.zeros((3,7), dtype = np.int)

data = np.zeros((Xt.shape[0],7), dtype=np.float)

rr = 0
for row in Xt:
#    print(row)
    data[rr,:] = np.array([dict_buying[row[0]], dict_maint[row[1]], 
           dict_doors[row[2]], dict_persons[row[3]], dict_lug_boot[row[4]],
           dict_safety[row[5]], dict_class[row[6]]])
    rr = rr + 1
    
    
# Part 2: training set = 1210 & test set = 518

np.random.seed(10)
set_perm = np.random.permutation(range(data.shape[0]))
ind_train = set_perm[0:1210]
ind_test = set_perm[1210:]
    
X_train = data[ind_train,0:6]
y_train = data[ind_train,6]

X_test = data[ind_test,0:6]
y_test = data[ind_test,6]

# Part 3: standardization

from sklearn import preprocessing

scaler = preprocessing.StandardScaler().fit(X_train)
#print('X scaler mean: ', scaler.mean_)
#print('X scaler variance: ', scaler.var_)

standardized_X = scaler.transform(X_train)

# Part 4: feature selection

from sklearn.ensemble import ExtraTreesClassifier
slct = ExtraTreesClassifier()
slct.fit(X_train, y_train)
# display the relative importance of each attribute
print('\n\nfeature_importances:')
print(slct.feature_importances_)


# Part 5: decision tree classifier

from sklearn import tree
#from sklearn.tree import DecisionTreeClassifier
# fit a ID3 model to the data : 'entropy', or CART model: 'gini'
# change max_depth & min_samples_leaf to simplify the tree
clf = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best')

#clf.fit(standardized_X, y_train)
clf.fit(X_train, y_train)
print('\n\nDecision_tree_classifier:')
print(clf)

from sklearn import metrics
std_X_test = scaler.transform(X_test)
expected_clf = y_test
#predicted_clf = clf.predict(std_X_test)
predicted_clf = clf.predict(X_test)
# summarize the fit of the model
print('\n\nDecision_tree_classifier_prediction_summary:')
print(metrics.classification_report(expected_clf, predicted_clf))
print(metrics.confusion_matrix(expected_clf, predicted_clf))

# visualization
# D:\Anaconda3\Library\bin\graphviz
import graphviz
dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'],  
                         class_names=['unacc', 'acc', 'good', 'vgood'],  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)
graph.render("cars_tree")

# Part 6: Naive Bayes classifier

from sklearn.naive_bayes import GaussianNB
nby = GaussianNB()
nby.fit(X_train, y_train)
print('\n\nNaive_bayes_classifier:')
print(nby)
# make predictions
expected_nby = y_test
predicted_nby = nby.predict(X_test)
# summarize the fit of the model
print('\n\nNaive_bayes_classifier_prediction_summary:')
print(metrics.classification_report(expected_nby, predicted_nby))
print(metrics.confusion_matrix(expected_nby, predicted_nby))

# Part 7: k Nearest Neighbors classifier

from sklearn.neighbors import KNeighborsClassifier
# fit a k-nearest neighbor model to the data
knb = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
                           weights='uniform')
knb.fit(X_train, y_train)
print('\n\nK_nearest_neighbors_classifier:')
print(knb)
# make predictions
expected_knb = y_test
predicted_knb = knb.predict(X_test)
# summarize the fit of the model
print('\n\nK_neighbors_classifier_prediction_summary:')
print(metrics.classification_report(expected_knb, predicted_knb))
print(metrics.confusion_matrix(expected_knb, predicted_knb))

# Part 8: support vector machine classifier

from sklearn.svm import SVC
# fit a SVM model to the data
svmc = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
           decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
           max_iter=-1, probability=False, random_state=None, shrinking=True,
           tol=0.001, verbose=False)
svmc.fit(X_train, y_train)
print('\n\nSVM_classifier:')
print(svmc)
# make predictions
expected_svmc = y_test
predicted_svmc = svmc.predict(X_test)
# summarize the fit of the model
print('\n\nSVM_classifier_prediction_summary:')
print(metrics.classification_report(expected_svmc, predicted_svmc))
print(metrics.confusion_matrix(expected_svmc, predicted_svmc))

# Part 9: Multilayer Peceptron classifier

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 3), random_state=1)
mlp.fit(X_train, y_train)
print('\n\nMLP_classifier:')
print(mlp)
# make predictions
expected_mlp = y_test
predicted_mlp = mlp.predict(X_test)
# summarize the fit of the model
print('\n\nMLP_classifier_prediction_summary:')
print(metrics.classification_report(expected_mlp, predicted_mlp))
print(metrics.confusion_matrix(expected_mlp, predicted_mlp))