# Final Project: Fraud in Enron Mail Data set


# Enron Data Set Project
##Question 1
**Summarize for us the goal of this project and how machine learning is useful in
 trying to accomplish it.As part of your answer, give some background on the 
 dataset and how it can be used to answer the project question. Were there any 
 outliers in the data when you got it, and how did you handle those?**
[relevant rubric items: “data exploration”, “outlier investigation”]  

The [Enron scandal](https://en.wikipedia.org/wiki/Enron_scandal) was a financial scandal that eventually led to the bankruptcy of the Enron Corporation, 
an American energy company based in Houston, Texas, and the de facto dissolution of Arthur Andersen, 
which was one of the five largest audit and accountancy partnerships in the world. 
In addition to being the largest bankruptcy reorganization in American history at that time, 
Enron was cited as the biggest audit failure.
Using the data set described below, the goal is to build a classifier in order
to predict whether a person is a person of interest (**POI**) or not.  

A person of interest is a person who fulfills one the following conditions:
- indicted
- settled without admitting guilt
- testified in exchange for immunity

Our data set is built on **Enron Email Dataset**, the **Enron 
statement of financial affairs**, and a **initial list of POIs**.  

####Enron Data Set 
 The [Enron Email Dataset](https://www.cs.cmu.edu/~./enron/) was collected and prepared by
  the CALO Project (A Cognitive Assistant that Learns and Organizes). 
It contains data from about 150 users, mostly senior management of Enron, 
organized into folders. The corpus contains a total of about 0.5M messages. 
This data was originally made public, and posted to the web, by the Federal Energy Regulatory 
Commission during its investigation.  
####Enron statemment of financial affairs
The [Enron statement of financial affairs](http://news.findlaw.com/hdocs/docs/enron/enron61702insiderpay.pdf)
consists in a pdf report including financial information such as salary, bonus received, 
exercised stock options...for different Enron employees. This pdf was parsed using a script
available in the udacity teaching materials
####Inital list of persons of interest
This list is part of the udacity teaching resources. It was populated thanks to reading news. Note 
that it lists persons who were employees at Enron and some who were not.

####Preprocessing
As preprocessing to this project, Udacity combined the Enron email and 
financial data into a dictionary, where each key-value pair in the dictionary
corresponds to one person. The dictionary key is the person's name, and the
value is another dictionary, which contains the names of all the features and
their values for that person. The features in the data fall into three major
types, namely financial features, email features and POI labels.

The *featureFormat* function converts this dictionary into numpy arrays and
offers the possibility to convert 'NaN' values into null values, and then remove
those null values (see function docstring).  
The *targetFeatureSplit* function splits the array returned from *featureFormat*
into 2 separate lists of features and labels.  
This features are then listed in alphabetical order by the
FeatureFormat function. So especially for the first person ALLEN, PHILLIP K,
the features are in this order:  
>{'bonus': 4175000,  
 'deferral_payments': 2869717,  
 'deferred_income': -3081055,  
 'director_fees': 'NaN',  
 'email_address': 'phillip.allen@enron.com',  
 'exercised_stock_options': 1729541,  
 'expenses': 13868,  
 'from_messages': 2195,  
 'from_poi_to_this_person': 47,  
 'from_this_person_to_poi': 65,  
 'loan_advances': 'NaN',  
 'long_term_incentive': 304805,  
 'other': 152,  
 'poi': False,  
 'ratio_from_poi_to_this_person': 0.0214123006833713,  
 'ratio_from_this_person_to_poi': 0.022398345968297727,  
 'restricted_stock': 126027,    
 'restricted_stock_deferred': -126027,  
 'salary': 201955,  
 'shared_receipt_with_poi': 1407,  
 'to_messages': 2902,  
 'total_payments': 4484442,  
 'total_stock_value': 1729541}

Machine learning will be useful to reach our objective because the volume of
information which shall drive the classification is very important:
    - number of people to 'screen': 150 people in the Enron Data set.
    - number of features: 14 financial features from the financial statements only, to which we 
    add 7 email features from the email dataset.  
    
Relying only on intuition and manual analysis to predict whether a person is a POI 
can't possibly provide us with optimal results. Thanks to machine learning we can take 
benefit of the big volume of data to ensure that we select our 'best' (will be discussed later)
features, and to train a classifier (specific algorithm and parameters 
still to be chosen) which will identify patterns in the data that a human would have 
difficultly spot. Therefore it should hopefully be able to make much better and faster 
predictions.

####Data Exploration
Thanks to the *info* function, I learnt for instance that there are:
- 18 POI in the dataset
    - 14 have non-null mail features
    - 17 have a non-null 'salary' value
    - 16 have a non-null 'bonus' value 
- 126 non POI
    - 72 have non-null mail features
    - 65 have a non-null 'salary' value
    - 77 have a non-null 'bonus' value 

####Outliers
Thanks to the max statistics returned by *info(details='stats')*, I could easily spot outliers.  Indeed the financial 
statements pdf includes a 'TOTAL'. The corresponding point has to be removed from the dictionary.
I removed an another point which is not a person: "the travel agency in the park" (also easily spot because this line 
is just above the last line 'TOTAL' in the pdf).
##Question 2
**What features did you end up using in your POI identifier, and what selection
process did you use to pick them? Did you have to do any scaling? Why or why not?
 As part of the assignment, you should attempt to engineer your own feature that
 does not come ready-made in the dataset -- explain what feature you tried to
 make, and the rationale behind it. (You do not necessarily have to use it in
 the final analysis, only engineer and test it.) In your feature selection step,
 if you used an algorithm like a decision tree, please also give the feature
 importances of the features that you use, and if you used an automated feature
 selection function like SelectKBest, please report the feature scores and
 reasons for your choice of parameter values.**   
[relevant rubric items: “create new features”, “intelligently select features”,
 “properly scale features”]  
 My hunch is that the following features shall be among the ones containing the most " information" and that they should
  make a good classifier:
- financial features
    - bonus
    - exercised_stock_options
    - salary
    - mail features:
    - from_poi_to_this_person
    - from_this_person_to_poi
    - shared_receipt_with_poi  
    
To confirm my intuition and also to ensure that I features with the most importance, I use
use selectKBest.  

I will also recreate the 2 features that we discussed within the course:    
- ratio of from_poi_to_this_person over from_messages, noted *ratio_from_poi_to_this_person*
- ratio of from_this_person_to_poi over to_messages, noted *ratio_from_this_person_to_poi*

The function *select_features* indicates me that 7 features with the biggest scores (the most "powerful") are:
>exercised_stock_options ->score: 24.8150797332
to_messages ->score: 24.1828986786
bonus ->score: 20.7922520472
restricted_stock ->score: 18.2896840434
deferred_income ->score: 11.4584765793
long_term_incentive ->score: 9.92218601319
ratio_from_poi_to_this_person ->score: 9.21281062198

Comments on the *f_classif* function:  
See [sklearn doc](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html#sklearn.feature_selection.f_classif)
and [explanation of ANOVA & F-test](http://blog.minitab.com/blog/adventures-in-statistics-2/understanding-analysis-of-variance-anova-and-the-f-test)
*f_classif computes the ANOVA F-value for the provided sample.  
ANOVA uses the F-test to determine whether the variability between group means is larger than the variability of the 
observations within the groups. If that ratio is sufficiently large, you can conclude that not all the means are equal.  
So it is a way to measure a feature’s relation to the response variable.  
The higher the F-statistic , the more different the group means, the more powerful the feature.   
##Question 3
**What algorithm did you end up using? What other one(s) did you try? How did 
model performance differ between algorithms?**  
[relevant rubric item: “pick an algorithm”]  
I want to try and compare the following algorithms:  
- Naive Bayes
- Decision Tree
- Random Forest (I already know random forest will perform better than Decision 
 Tree due to its design, but I'd still like to evaluate this)  
 
 I chose those algorithms, because I know that they are not affected by 
 feature scaling. So it will save me having to perform such preprocessing 
 operation.
 
 As suggested in the Udacity project notes, for more clarity and convenience,
 I will use *Pipeline* to perform the following processing steps:
1. Select features - 'select'
2. Create classifier - 'classifier'
    - nb - naive bayes
    - dt - decision tree
    - rdf - random forest

First I created classifiers using their default parameters.
The function *try_classifier* indicates me how model performance differ between algorithms:
>Accuracy of GaussianNB(): **0.83 (+/- 0.20)**  
Accuracy of DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='best'): **0.79 (+/- 0.19)**  
Accuracy of RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False): **0.85 (+/- 0.16)**  

##Question 4
**What does it mean to tune the parameters of an algorithm, and what can happen if you don’t
 do this well?  How did you tune the parameters of your particular algorithm? 
 What parameters did you tune? (Some algorithms do not have parameters that you need to tune
  -- if this is the case for the one you picked, identify and briefly explain how you would
   have done it for the model that was not your final choice or a different model that does 
   utilize parameter tuning, e.g. a decision tree classifier).**    
   [relevant rubric items: “discuss parameter tuning”, “tune the algorithm”]
Tuning the parameters of an algorithm is all about finding the right spot between Bias and Variance in order to have
the best model possible.  Algorithm parameters can influence their bias and/or variance.  
A high bias means that my model is oversimplified and doesn't pay attention to data --> can happen if I select a value
small for the k parameter in SelectKBest.  
A high variance means the opposite, that my model pays too much attention to data or is overfitted --> can happen if my 
algorithm is too optimzed: e.g. too many nodes (min_samples_split too small).     
At first I wanted to play also with the f_score function parameter of SelectKBest and consider both chi2 and f_classif, 
but there are some negative values among the financial features (e.g. *deferred_income*, *restricted_stock_deferred*).
So in the end I did not consider chi2 to avoid having to convert these features to positive 
values or having to ignore them.  
I finally tuned the following parameters:  
- SelectKBest
    - k
- Decision Tree
    - criterion
    - min samples split
    - splitter
-Random Forest
    - criterion
    - min samples split
    - number of trees (n_estimators)
I performed this tuning using *GridSearchCV* (see *tune* function).  
i largely inspired myself from this [code](http://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html)
to build my function. 
Here are the best score and best parameters set after tuning the random forest algorithm: 
>Best score: 0.881  
Best parameters set:  
	classify__criterion: 'gini'  
	classify__min_samples_split: 3  
	classify__n_estimators: 10  
	select_features__k: 7  
	select_features__score_func: <function f_classif at 0x05724630>  

##Question 5
**What is validation, and what’s a classic mistake you can make if you do it wrong? 
How did you validate your analysis?**    
[relevant rubric items: “discuss validation”, “validation strategy”]  
Validation consists in estimating model/algorithm performance, especially by splitting the data set in a training set and
a testing set, in order to be able to state how well one model has been trained, and how good it is good at predicting.
As I used GridSearchCV, a cross validation technique was already applied. As explained in the GridSearchCV [documentation](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html):  
>cv parameter:  
For integer/None inputs, if the estimator is a classifier and y is either binary or multiclass, StratifiedKFold is used.
In all other cases, KFold is used.
>
So in my case KFold was used: k learning experiments using k different train/test sets were run. Then the test results were averaged.

##Question 6
**Give at least 2 evaluation metrics and your average performance for each of them.  
Explain an interpretation of your metrics that says something human-understandable about your 
algorithm’s performance.**  
We usually consider precision and recall as evaulation metrics.
- precision: probability of a given prediction to be true. ie out of all predicted labels, how many were correct.
- recall: probability of class to be correctly predicted. e.g. out of how of all predicted POI, how many are really POIs.
Here are my metrics:
>
               precision    recall  f1-score   support

          0.0       0.88      1.00      0.94        38
          1.0       0.00      0.00      0.00         5
    avg/total       0.78      0.88      0.83        43
>


[relevant rubric item: “usage of evaluation metrics”]
