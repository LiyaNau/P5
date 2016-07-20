#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")


from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score

# features_list = ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 'salary', 'deferred_income']
features_list_all = ['salary',"bonus","to_messages",'total_payments','exercised_stock_options',
                 'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 'expenses',
                 'loan_advances', 'from_messages', 'other', 'from_this_person_to_poi', 'director_fees',
                'deferred_income', 'long_term_incentive', 'from_poi_to_this_person']
features_list = ["poi"]
features_list += features_list_all
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


data_dict.pop('TOTAL')
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# Provided to give you a starting point. Try a variety of classifiers.
# from sklearn.cross_validation import train_test_split
# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedShuffleSplit

cv = StratifiedShuffleSplit(labels,n_iter = 30, test_size=0.2)
clf = RandomForestClassifier(class_weight="balanced")

param_grid = {'n_estimators':[10,50,100,200],
              'max_features':[10,5,2]}
grid = GridSearchCV(clf, param_grid ,verbose=True, cv=cv, scoring = 'recall' )
grid.fit(features,labels)
print "best estimator:", grid.best_estimator_
print "best score:", grid.best_score_
clf = grid.best_estimator_


### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

#clf.fit(features_train,labels_train)
#print clf.oob_score


dump_classifier_and_data(clf, my_dataset, features_list)