# Final Project: Fraud in Enron Mail Data set

##Introduction
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
##Data set
Our data set is built on **Enron Email Dataset**, the **Enron 
statement of financial affairs**, and a **initial list of POIs**.  

###Enron Data Set 
 The [Enron Email Dataset](https://www.cs.cmu.edu/~./enron/) was collected and prepared by
  the CALO Project (A Cognitive Assistant that Learns and Organizes). 
It contains data from about 150 users, mostly senior management of Enron, 
organized into folders. The corpus contains a total of about 0.5M messages. 
This data was originally made public, and posted to the web, by the Federal Energy Regulatory 
Commission during its investigation.  
###Enron statemment of financial affairs
The [Enron statement of financial affairs](http://news.findlaw.com/hdocs/docs/enron/enron61702insiderpay.pdf)
consists in a pdf report including financial information such as salary, bonus received, 
exercised stock options...for different Enron employees. This pdf was parsed using a script
available in the udacity teaching materials
###Inital list of persons of interest
This list is part of the udacity teaching resources. It was populated thanks to reading news. Note 
that it lists persons who were employees at Enron and some who were not.


##Questions
1. *Summarize for us the goal of this project and how machine learning is useful in trying to 
accomplish it. As part of your answer, give some background on the dataset and how it can be 
used to answer the project question. Were there any outliers in the data when you got it, 
and how did you handle those?*
[relevant rubric items: “data exploration”, “outlier investigation”]  
Machine learning is useful to reach our objective because the volume of information which 
shall drive the classification is very important:
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

2. *What features did you end up using in your POI identifier, and what selection process did 
you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment,
 you should attempt to engineer your own feature that does not come ready-made in the dataset
  -- explain what feature you tried to make, and the rationale behind it. 
  (You do not necessarily have to use it in the final analysis, only engineer and test it.) 
  In your feature selection step, if you used an algorithm like a decision tree, please also 
  give the feature importances of the features that you use, and if you used an automated 
  feature selection function like SelectKBest, please report the feature scores and reasons for
   your choice of parameter values.* 
   [relevant rubric items: “create new features”, “intelligently select features”, “properly scale features”]

3. What algorithm did you end up using? What other one(s) did you try? How did model 
performance differ between algorithms?  
[relevant rubric item: “pick an algorithm”]

4. What does it mean to tune the parameters of an algorithm, and what can happen if you don’t
 do this well?  How did you tune the parameters of your particular algorithm? 
 What parameters did you tune? (Some algorithms do not have parameters that you need to tune
  -- if this is the case for the one you picked, identify and briefly explain how you would
   have done it for the model that was not your final choice or a different model that does 
   utilize parameter tuning, e.g. a decision tree classifier).  
   [relevant rubric items: “discuss parameter tuning”, “tune the algorithm”]

5. What is validation, and what’s a classic mistake you can make if you do it wrong? 
How did you validate your analysis?  
[relevant rubric items: “discuss validation”, “validation strategy”]

6. Give at least 2 evaluation metrics and your average performance for each of them.  
Explain an interpretation of your metrics that says something human-understandable about your 
algorithm’s performance. 
[relevant rubric item: “usage of evaluation metrics”]


##Features

##Algorithm

##Evaluation
