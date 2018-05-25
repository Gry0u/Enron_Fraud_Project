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
from sklearn.metrics import classification_report
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Task 1: Select what features you'll use.
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".


def info(details=None):
    """
    Prints depending on details parameter:
     - the features in the dataset.
     - value of all features for a selected person
     - values types details
     - dataset statistics

    :param details:
        - person name in the form [LASTNAME FIRSTNAME (MIDDLENAME INITIAL)]
        - or value from ['all, 'poi, 'non_poi] to get values types details
        - or 'features' to list features alphabetically
        - or 'stats' to get stats on all features
    :return: list of features names
    """
    #create dataframe
    data_df = pd.DataFrame(data_dict)
    data_df = data_df.transpose()

    #list features
    features_name_list = []
    features_name_list = [col for col in data_df.columns if col not in ['email_address', 'poi']]
    features_name_list = ['poi'] + features_name_list
    #force values types
    numeric_features = [col for col in data_df.columns if col not in ['poi', 'email_address']]
    for col in numeric_features:
        data_df[col] = data_df[col].apply(lambda c: pd.to_numeric(c, errors='coerce'))

    if details == 'all':
        pprint(data_df.info(verbose=True))
        #in this case the features are already counted and listed by .info()
        list_features = False
        number_features = False
    elif details == 'poi':
        pprint(data_df[data_df['poi'] == True].info(verbose=True))
        list_features = False
        number_features = False
    elif details =='non_poi':
        pprint(data_df[data_df['poi'] == False].info(verbose=True))
        list_features = False
        number_features = False
    elif details in data_dict.keys():
        pprint(my_dataset[person])
    elif details == 'features':
        print'features:'
        pprint(sorted([feature for feature in features_name_list if feature != 'poi']))
        print
        print 'number of features:', len([feature for feature in features_name_list if feature != 'poi'])
    elif details == 'stats':
        print data_df.describe()
    elif details is None:
        pass
    else:
        'the "details" parameter can take only the followings values:"[PERSON NAME], "features","all", "poi", "non_poi", "stats'
    return features_name_list


#info(details='poi')
#info(details='non_poi')
#info(details='all')
#info('ALLEN PHILLIP K')
#info(details='stats')

features_list=info()

# Task 2: Remove outliers
data_dict.pop('TOTAL',0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')

# Task 3: Create new feature(s)
# Store to my_dataset for easy export below.
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

#add created features to the features list
features_list = features_list + ['ratio_from_this_person_to_poi', 'ratio_from_poi_to_this_person']

### Impute missing email features to mean
email_features = ['to_messages', 'from_poi_to_this_person', 'ratio_from_poi_to_this_person', 'from_messages',
                  'from_this_person_to_poi','ratio_from_this_person_to_poi', 'shared_receipt_with_poi']
from collections import defaultdict
email_feature_sums = defaultdict(lambda:0)
email_feature_counts = defaultdict(lambda:0)

for employee, features in data_dict.iteritems():
    for ef in email_features:
        if features[ef] != "NaN":
            email_feature_sums[ef] += features[ef]
            email_feature_counts[ef] += 1

email_feature_means = {}
for ef in email_features:
    email_feature_means[ef] = float(email_feature_sums[ef]) / float(email_feature_counts[ef])

for employee, features in data_dict.iteritems():
    for ef in email_features:
        if features[ef] == "NaN":
            features[ef] = email_feature_means[ef]

# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

#Select best features from 21


def select_features():
    selector = SelectKBest()
    selector.fit(features, labels)
    zipped = zip(info()[1:], selector.scores_)
    zipped.sort(key= lambda x:x[1], reverse=True)
    for f, s in zipped:
        print f,'->score:', s
    return

#select_features()

# Task 4: Try a varity of classifiers
# Please name your classifier clf for easy export below.
# Note that if you want to do PCA or other multi-stage operations,
# you'll need to use Pipelines. For more info:
# http://scikit-learn.org/stable/modules/pipeline.html

def try_classifier(selector, classifier, X=features, y=labels, print_accuracy=False):
    #create pipeline
    model = Pipeline([('select_features',selector), ('classify', classifier)])
    #evaluate pipeline
    kfold = KFold(len(labels), n_folds=10, random_state=9)
    scores = cross_val_score(model, X, y, cv=kfold)
    if print_accuracy:
        print 'Accuracy of %s: %0.2f (+/- %0.2f)' % (classifier, scores.mean(), scores.std() *2)
    return model


#classifiers = [GaussianNB(), DecisionTreeClassifier(), RandomForestClassifier()]
#for classifier in classifiers:
    #try_classifier(SelectKBest(), classifier, print_accuracy=True)


# Task 5: Tune your classifier to achieve better than .3 precision and recall using our testing script.
# Check the tester.py script in the final projectfolder for details on the evaluation method, especially the
# test_classifier function. Because of the small size of the dataset, the script uses stratified shuffle split
# cross validation. For more info:
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
param_KBest = {'select_features__score_func':[f_classif],'select_features__k':[6, 7, 8, 9, 10, 11, 12, 13]}
params_dt = {'classify__criterion':['gini', 'entropy'],'classify__splitter':['best', 'random'],
             'classify__min_samples_split':[2, 3, 4, 5, 10]}
params_rdf = {'classify__n_estimators':[3,5,10,20,40], 'classify__criterion':['gini', 'entropy'],
              'classify__min_samples_split':[2,3,4,5,10]}


def tune(selector, classifier, X=features, y=labels, print_workflow=False, print_best=False):
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

    if print_workflow:
        grid_search = GridSearchCV(try_classifier(selector, classifier, X, y), parameters, verbose=1)
        print "Performing grid search..."
        print "pipeline:", [name for name, _ in try_classifier(selector, classifier, X, y).steps]
        print "parameters:"
        pprint(parameters)
        t0 = time()
        grid_search.fit(X, y)
        print "done in %0.3fs" % (time() - t0)
        print
    else:
        grid_search = GridSearchCV(try_classifier(selector, classifier, X, y), parameters)
        grid_search.fit(X, y)
    if print_best:
        print "Best score: %0.3f" % grid_search.best_score_
        print "Best parameters set:"
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print "\t%s: %r" % (param_name, best_parameters[param_name])
    return grid_search.fit(X, y)

#tune(SelectKBest(), RandomForestClassifier(), print_best=True)


# Task 6: evaluation metrics
features_train, features_test, labels_train, labels_test = \
train_test_split(features, labels, test_size=0.3, random_state=42)
pln=try_classifier(SelectKBest(k=5), RandomForestClassifier(n_estimators=10, criterion='entropy', min_samples_split=5))
clf = pln.fit(features_train, labels_train)
#print classification_report(labels_test, clf.predict(features_test))
# Task 7: Dump your classifier, dataset, and features_list so anyone can
# check your results. You do not need to change anything below, but make sure
# that the version of poi_id.py that you submit can be run on its own and
# generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)