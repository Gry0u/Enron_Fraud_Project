#!/usr/bin/python

import sys
import os
import pickle
import pandas as pd
from pprint import pprint
from time import time
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.cross_validation import train_test_split, KFold, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, precision_score, recall_score
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features
##I am defining a dataframe because I found it convenient/fast to get all the features names
data_df = pd.DataFrame(data_dict)
data_df = data_df.transpose()
features_name_list=[col for col in data_df.columns if col not in ['email_address','poi']]
features_name_list=['poi']+features_name_list
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Task 2: Remove outliers
data_dict.pop('TOTAL',0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

#add new features & convert negative values to positive
for person, features_person in my_dataset.iteritems():
    from_poi = features_person['from_poi_to_this_person']
    to_poi = features_person['from_this_person_to_poi']
    from_ = features_person['from_messages']
    to_ = features_person['to_messages']
    #deferred_income = features_person['deferred_income']
    #deferred_stock = features_person['restricted_stock_deferred']
    if from_poi == 'NaN' or from_ == 'NaN':
        features_person['ratio_from_poi_to_this_person'] = 'NaN'
    else:
        features_person['ratio_from_poi_to_this_person'] = float(from_poi)/float(from_)
    if to_poi == 'NaN' or to_ == 'NaN':
        features_person['ratio_from_this_person_to_poi'] = 'NaN'
    else:
        features_person['ratio_from_this_person_to_poi'] = float(to_poi) / float(to_)
    if features_person['deferred_income'] != 'NaN':
        features_person['deferred_income'] = abs(features_person['deferred_income'])
       # print person, deferred_income
    if features_person['restricted_stock_deferred'] != 'NaN':
        features_person['restricted_stock_deferred'] = abs(features_person['restricted_stock_deferred'])

#visualize one data point:
pprint(my_dataset['ALLEN PHILLIP K'])
print

features_name_list = features_name_list + ['ratio_from_this_person_to_poi', 'ratio_from_poi_to_this_person']
#check order of features to be able to identify them when they will be stored in a ndarray
print'features:'
pprint(sorted([feature for feature in features_name_list if feature!='poi']))
print
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_name_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

#print 'Number of features: ', len(features[0])
#Select best features from 19
selector = SelectKBest()
selector.fit(features, labels)
#print selector.scores_

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

def try_classifier(classifier, print_accuracy=False):
    #create pipeline
    model = Pipeline([('select_features',selector),('classify', classifier)])
    #evaluate pipeline
    kfold = KFold(len(labels), n_folds=10, random_state=9)
    scores = cross_val_score(model, features, labels, cv=kfold)
    if print_accuracy:
        print 'Accuracy of %s: %0.2f (+/- %0.2f)' % (classifier, scores.mean(), scores.std() *2)
    return model


classifiers = [GaussianNB(), DecisionTreeClassifier(), RandomForestClassifier()]
#for classifier in classifiers:
    #try_classifier(classifier)


# Task 5: Tune your classifier to achieve better than .3 precision and recall using our testing script.
# Check the tester.py script in the final projectfolder for details on the evaluation method, especially the
# test_classifier function. Because of the small size of the dataset, the script uses stratified shuffle split
# cross validation. For more info:
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
param_KBest = {'select_features__score_func':[f_classif, chi2],'select_features__k':[3, 4, 5, 6, 7, 8, 9, 10]}
params_dt = {'classify__criterion':['gini', 'entropy'],'classify__splitter':['best', 'random'],
             'classify__min_samples_split':[2, 3, 4, 5, 10]}
params_rdf = {'classify__n_estimators':[3,5,10,20,40], 'classify__criterion':['gini', 'entropy'],
              'classify__min_samples_split':[2,3,4,5,10]}

#http://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html
#note about cross validation in gridsearchcv: cv param: For integer/None inputs, if the estimator is a classifier and y
#  is either binary or multiclass, StratifiedKFold is used. In all other cases, KFold is used. --> so in my case
#StratifiedKFold is used.

def tune(classifier, print_best=False):
    if isinstance(classifier, GaussianNB):
        parameters = dict(param_KBest)
    elif isinstance(classifier, DecisionTreeClassifier):
        parameters = dict(param_KBest)
        parameters.update(params_dt)
    elif isinstance(classifier, RandomForestClassifier):
        parameters = dict(param_KBest)
        parameters.update(params_rdf)
    else:
        'The dictionary for this dictionary hasn"t been defined!'

    grid_search = GridSearchCV(try_classifier(classifier), parameters, verbose=1)
    print "Performing grid search..."
    print "pipeline:", [name for name, _ in try_classifier(classifier).steps]
    print "parameters:"
    pprint(parameters)
    t0 = time()
    grid_search.fit(features, labels)
    print "done in %0.3fs" % (time() - t0)
    if print_best:
        print
        print "Best score: %0.3f" % grid_search.best_score_
        print "Best parameters set:"
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print "\t%s: %r" % (param_name, best_parameters[param_name])
    return grid_search.fit(features, labels)


#tune(DecisionTreeClassifier())

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)