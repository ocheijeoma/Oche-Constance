#!/usr/bin/env python
# coding: utf-8

# In[179]:


# Library to suppress warnings or deprecation notes
import warnings
warnings.filterwarnings("ignore")

# Importing Libraries to help with reading data (tabular and numerical)

import pandas as pd
import numpy as np

# Importing Libraries to perform statistical analysis
import scipy.stats as stats
import sklearn

# Importing Library to split data
from sklearn.model_selection import train_test_split

# Importing libraries to help with data visualization 
import matplotlib.pyplot as plt
import seaborn as sns

# Removing the limit for the number of displayed columns
pd.set_option("display.max_columns", None)
# Sets the limit for the number of displayed rows
pd.set_option("display.max_rows", 200)

# statemodels
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant

# Importing library To build logistic regression model
from sklearn.linear_model import LogisticRegression

# Importing libraries To get diferent metric scores
from sklearn import metrics
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    make_scorer,
)

# Importing Library to build decision tree classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Importing Library to import different ensemble classifiers
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier

# Importing library To tune different models
from sklearn.model_selection import GridSearchCV




# ## *Importing Dataset*

# In[180]:


# Loading the survey into notebook
mydata = pd.read_csv('D:\survey.csv')


# In[181]:


# Checking the dimensions of the dataset
mydata.shape


# In[182]:


# Taking a peek at the first five entries in the dataset
mydata.head(20)


# ## *Data Pre-processing*

# ### Missing Values Detection and Remdiation

# In[183]:


# Checking if there are missing values
mydata.isna().apply(pd.value_counts).T


# In[184]:


# Checking data types of each attribute
mydata.info()


# In[185]:


mhdata.duplicated().any()


# In[186]:


mydata.duplicated().any()


# In[187]:


# Removing unnecessary columns that are not required for exploratory data analysis (EDA) and modeling 
mydata.drop(columns=['Timestamp', 'Country', 'state', 'comments'], inplace = True)


# In[188]:


# Changing column names to create consistency and enhance comprehension
mydata.rename({'self_employed' : 'Self_Employed', 'family_history' : 'Family_History', 
           'treatment' : 'Treatment', 'work_interfere' : 'Work_Interfere', 
           'no_employees': 'Employee_Count_Company', 'remote_work': 'Remote_Work', 'tech_company': 'Tech_Company', 
           'benefits': 'Benefits', 'care_options': 'Care_Options', 'wellness_program': 'Wellness_Program', 
           'seek_help': 'Seek_Help', 'anonymity': 'Anonymity', 'leave': 'Medical_Leave', 
           'mental_health_consequence': 'Mental_Health_Consequence', 
           'phys_health_consequence': 'Physical_Health_Consequence', 'coworkers': 'Coworkers_Reach', 
           'supervisor': 'Supervisor_Reach', 'mental_health_interview': 'Mental_Health_Interview', 
           'phys_health_interview': 'Physical_Health_Interview', 'mental_vs_physical': 'Mental_VS_Physical', 
           'obs_consequence': 'Observed_Consequence_Workplace'} , inplace = True , axis = 1)


# In[189]:


# Checking the entries for age
mydata['Age'].unique()


# In[190]:


# calculating the median age
median_age = mydata['Age'].median()
print(median_age)


# In[191]:


# replacing impossible values with the median age
mydata['Age'].replace([mydata['Age'][mydata['Age'] < 15]], median_age, inplace = True)
mydata['Age'].replace([mydata['Age'][mydata['Age'] > 100]], median_age, inplace = True)

mydata['Age'].unique()


# In[192]:


# Checking the entries for gender
mydata['Gender'].unique()


# In[193]:


# Analysis to three categories
mydata['Gender'].replace(['Male ', 'male', 'M', 'm', 'Male', 'Cis Male',
                     'Man', 'cis male', 'Mail', 'Male-ish', 'Male (CIS)',
                      'Cis Man', 'msle', 'Malr', 'Mal', 'maile', 'Make',], 'Male', inplace = True)

mydata['Gender'].replace(['Female ', 'female', 'F', 'f', 'Woman', 'Female',
                     'femail', 'Cis Female', 'cis-female/femme', 'Femake', 'Female (cis)',
                     'woman',], 'Female', inplace = True)

mydata["Gender"].replace(['Female (trans)', 'queer/she/they', 'non-binary',
                     'fluid', 'queer', 'Androgyne', 'Trans-female', 'male leaning androgynous',
                      'Agender', 'A little about you', 'Nah', 'All',
                      'ostensibly male, unsure what that really means',
                      'Genderqueer', 'Enby', 'p', 'Neuter', 'something kinda male?',
                      'Guy (-ish) ^_^', 'Trans woman',], 'Queer', inplace = True)


# In[194]:


mydata['Gender'].value_counts()


# In[195]:


# Limiting answers to boolean-like values
columns_to_print = ['Self_Employed', 'Family_History','Treatment', 'Work_Interfere', 'Employee_Count_Company', 'Remote_Work',
                    'Tech_Company', 'Benefits', 'Care_Options', 'Wellness_Program',
                    'Seek_Help', 'Anonymity', 'Medical_Leave', 'Mental_Health_Consequence',
                    'Physical_Health_Consequence', 'Coworkers_Reach', 'Supervisor_Reach',
                    'Mental_Health_Interview', 'Physical_Health_Interview', 'Mental_VS_Physical',
                    'Observed_Consequence_Workplace']

for column in columns_to_print:
    print(f"{column}:")
    print(mydata[column].value_counts())
    print()


# In[196]:


mydata['Treatment'] = np.where(mydata['Treatment'] == 'Yes', 1, 0)


# In[197]:


mydata.sample(10)


# ### *Variable Declaration*

# In[198]:


# separating dependent and independent variables.
X = mydata.drop(["Treatment"], axis=1)
Y = mydata["Treatment"]

# The independent variables will be transformed into dummy variables
X = pd.get_dummies(X, drop_first=True)

# adding constant. This is a requirement of Stats Model library. It creates a new column with float value 1
X = sm.add_constant(X)


# In[199]:


# X with the dummy variables
X.head()


# ### *Train-Test Split*

# In[200]:


# Splitting data in train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.30, random_state=1
)


# In[201]:


# Defining a function to compute different metrics to check performance of a classification model built using statsmodels
def model_performance_classification(model, predictors, target, threshold=0.5):
    """
    Function to compute different metrics to check classification model performance

    model: classifier
    predictors: independent variables
    target: dependent variable
    threshold: threshold for classifying the observation as class 1
    """

    # checking which probabilities are greater than threshold
    pred_temp = model.predict(predictors) > threshold
    # rounding off the above values to get classes
    pred = np.round(pred_temp)

    acc = accuracy_score(target, pred)  # to compute Accuracy
    recall = recall_score(target, pred)  # to compute Recall
    precision = precision_score(target, pred)  # to compute Precision
    f1 = f1_score(target, pred)  # to compute F1-score

    # creating a dataframe of metrics
    df_perf = pd.DataFrame(
        {
            "Accuracy": acc,
            "Recall": recall,
            "Precision": precision,
            "F1": f1,
        },
        index=[0],
    )

    return df_perf


# ## MODEL CREATION

# ### *Logistic regression model*

# In[202]:


# fitting logistic regression model
logit = sm.Logit(y_train, X_train.astype(float))
lg = logit.fit(disp=True)

print(lg.summary())


# ### *Plotting the Confusion Matrix For the Classification Model*

# In[203]:


# defining a function to plot the confusion_matrix of a classification model


def confusion_matrix_statsmodels(model, predictors, target, threshold=0.5):
    """
    To plot the confusion_matrix with percentages

    model: classifier
    predictors: independent variables
    target: dependent variable
    threshold: threshold for classifying the observation as class 1
    """
    y_pred = model.predict(predictors) > threshold
    cm = confusion_matrix(target, y_pred)
    group_names = ["True Negative", "False Positive", "False Negative", "True Positive"]
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)]
    labels = [
        f"{v1}\n{v2}\n{v3}"
        for v1, v2, v3 in zip(group_names, group_counts, group_percentages)
    ]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cm, annot=labels, fmt="", cmap="YlGnBu")
    



# In[204]:


print("Checking model performance on train set:")
confusion_matrix_statsmodels(lg, X_train, y_train, threshold=0.5)


# In[205]:


print("Training performance:")
model_performance_classification(lg, X_train, y_train)


# In[206]:


print("Checking model performance on test set:")
confusion_matrix_statsmodels(lg, X_test, y_test, threshold=0.5)


# In[207]:


print("Test performance:")
logistic_regression_perf = model_performance_classification(lg, X_test, y_test)
logistic_regression_perf


# In[208]:


logit_roc_auc_train = roc_auc_score(y_train, lg.predict(X_train))
fpr, tpr, thresholds = roc_curve(y_train, lg.predict(X_train))
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label="Logistic Regression (area = %0.2f)" % logit_roc_auc_train)
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic")
plt.legend(loc="lower right")
plt.show()


# In[209]:


# Finding the balanced threshold
y_scores = lg.predict(X_train)
prec, rec, tre = precision_recall_curve(
    y_train,
    y_scores,
)


def plot_prec_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="precision")
    plt.plot(thresholds, recalls[:-1], "g--", label="recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])


plt.figure(figsize=(10, 7))
plot_prec_recall_vs_threshold(prec, rec, tre)
plt.show()


# In[210]:


print("Training performance:")
model_performance_classification(lg, X_train, y_train, threshold=0.63)


# ## *Decision Tree Model*

# In[211]:


# Decision Tree Modeling
dTree = DecisionTreeClassifier(criterion="gini", random_state=1)
dTree.fit(X_train, y_train)


# In[212]:


dTree_model_train_perf = model_performance_classification(
    dTree, X_train, y_train
)
print("Training performance:\n", dTree_model_train_perf)
dTree_model_test_perf = model_performance_classification(dTree, X_test, y_test)
print("Testing performance:\n", dTree_model_test_perf)


# In[213]:


# function to create Confusion matrix
def create_confusion_matrix(model, predictors, target, figsize=(5, 5)):
    y_pred = model.predict(predictors)
    cm = confusion_matrix(target, y_pred)
    group_names = ["True Neg", "False Pos", "False Neg", "True Pos"]
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)]
    labels = [
        f"{v1}\n{v2}\n{v3}"
        for v1, v2, v3 in zip(group_names, group_counts, group_percentages)
    ]
    labels = np.asarray(labels).reshape(2, 2)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=labels, fmt="", cmap="YlGnBu")


# In[214]:


# Creating confusion matrix
create_confusion_matrix(dTree, X_test, y_test, figsize=(4, 3))


# In[215]:


feature_names = list(X.columns)
print(feature_names)


# In[216]:


print(tree.export_text(dTree, feature_names=feature_names, show_weights=False))


# In[217]:


plt.figure(figsize=(70, 60))
tree.plot_tree( dTree, feature_names=feature_names, filled=True, fontsize=9, node_ids=True,)
plt.show()


# In[218]:


# Printing feature importances
print(
    pd.DataFrame(
        dTree.feature_importances_, columns=["Imp"], index=X_train.columns
    ).sort_values(by="Imp", ascending=False)
)


# In[219]:


# Loading feature importance in a graph to get a better understanding
importances = dTree.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(13, 11))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="violet", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()


# In[220]:


new_respondent = pd.DataFrame([{
    'const': 1,                                 
    'Age': 27,                                      
    'Gender_Male': 0,                              
    'Gender_Queer': 0,                            
    'Self_Employed_Yes': 0,                      
    'Family_History_Yes': 1,                     
    'Work_Interfere_Often': 0,                     
    'Work_Interfere_Rarely': 0,                    
    'Work_Interfere_Sometimes': 1,            
    'Employee_Count_Company_25-Jun': 0,          
    'Employee_Count_Company_26-100': 1,             
    'Employee_Count_Company_500-1000': 0,           
    'Employee_Count_Company_5-Jan': 0,              
    'Employee_Count_Company_More than 1000': 0,     
    'Remote_Work_Yes': 1,                          
    'Tech_Company_Yes': 1,                        
    'Benefits_No': 0,                              
    'Benefits_Yes': 1,                            
    'Care_Options_Not sure': 0,                  
    'Care_Options_Yes': 1,                        
    'Wellness_Program_No': 1,                      
    'Wellness_Program_Yes': 0,                   
    'Seek_Help_No': 0,                          
    'Seek_Help_Yes': 1,                       
    'Anonymity_No': 0,                    
    'Anonymity_Yes': 1,                        
    'Medical_Leave_Somewhat difficult': 0,         
    'Medical_Leave_Somewhat easy': 0,             
    'Medical_Leave_Very difficult': 0,         
    'Medical_Leave_Very easy': 1,                
    'Mental_Health_Consequence_No': 0,           
    'Mental_Health_Consequence_Yes': 0,         
    'Physical_Health_Consequence_No': 0,          
    'Physical_Health_Consequence_Yes': 0,        
    'Coworkers_Reach_Some of them': 0,            
    'Coworkers_Reach_Yes': 1,                     
    'Supervisor_Reach_Some of them': 0,           
    'Supervisor_Reach_Yes': 0,                     
    'Mental_Health_Interview_No': 0,               
    'Mental_Health_Interview_Yes': 0,              
    'Physical_Health_Interview_No': 0,             
    'Physical_Health_Interview_Yes': 0,             
    'Mental_VS_Physical_No': 0,                    
    'Mental_VS_Physical_Yes': 1,                   
    'Observed_Consequence_Workplace_Yes': 0,

    
}],columns=[
    'const',                                 
    'Age',                                      
    'Gender_Male',                              
    'Gender_Queer',                           
    'Self_Employed_Yes',                      
    'Family_History_Yes',                     
    'Work_Interfere_Often',                     
    'Work_Interfere_Rarely',                    
    'Work_Interfere_Sometimes',            
    'Employee_Count_Company_25-Jun',          
    'Employee_Count_Company_26-100',             
    'Employee_Count_Company_500-1000',           
    'Employee_Count_Company_5-Jan',              
    'Employee_Count_Company_More than 1000',     
    'Remote_Work_Yes',                         
    'Tech_Company_Yes',                        
    'Benefits_No',                              
    'Benefits_Yes',                            
    'Care_Options_Not sure',                  
    'Care_Options_Yes',                       
    'Wellness_Program_No',                     
    'Wellness_Program_Yes',                   
    'Seek_Help_No',                          
    'Seek_Help_Yes',                       
    'Anonymity_No',                    
    'Anonymity_Yes',                        
    'Medical_Leave_Somewhat difficult',         
    'Medical_Leave_Somewhat easy',             
    'Medical_Leave_Very difficult',         
    'Medical_Leave_Very easy',                
    'Mental_Health_Consequence_No',           
    'Mental_Health_Consequence_Yes',          
    'Physical_Health_Consequence_No',          
    'Physical_Health_Consequence_Yes',        
    'Coworkers_Reach_Some of them',            
    'Coworkers_Reach_Yes',                     
    'Supervisor_Reach_Some of them',           
    'Supervisor_Reach_Yes',                     
    'Mental_Health_Interview_No',               
    'Mental_Health_Interview_Yes',              
    'Physical_Health_Interview_No',             
    'Physical_Health_Interview_Yes',             
    'Mental_VS_Physical_No',                    
    'Mental_VS_Physical_Yes',                   
    'Observed_Consequence_Workplace_Yes',],)
new_respondent


# In[225]:


predictionlg = lg.predict(new_respondent)
print('Prediction:', round(predictionlg[0]))


# In[226]:


# testing performance comparison

models_test_comp_df = pd.concat(
    [
        logistic_regression_perf.T,
        dTree_model_test_perf.T,
    ],
    axis=1,
)
models_test_comp_df.columns = [
    "Logistic Regression",
    "Decision Tree",
   
]
print("Testing performance comparison:")
models_test_comp_df


# ## *Random Forest*

# In[227]:


forest = RandomForestClassifier(criterion="gini", random_state=0)
forest.fit(X_train, y_train)


# In[228]:


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[229]:


forest_model_train_perf = model_performance_classification(
    forest, X_train, y_train
)
print("Training performance:\n", forest_model_train_perf)
forest_model_test_perf = model_performance_classification(forest, X_test, y_test)
print("Testing performance:\n", forest_model_test_perf)


# In[230]:


print(forest.score(X_test,y_test))
#print(forest.score(X_train, y_train))


# In[231]:


# function to create Confusion matrix for Random forest
def create_confusion_matrix(model, predictors, target, figsize=(5, 5)):
    y_pred = model.predict(predictors)
    cm = confusion_matrix(target, y_pred)
    group_names = ["True Neg", "False Pos", "False Neg", "True Pos"]
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)]
    labels = [
        f"{v1}\n{v2}\n{v3}"
        for v1, v2, v3 in zip(group_names, group_counts, group_percentages)
    ]
    labels = np.asarray(labels).reshape(2, 2)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=labels, fmt="", cmap="YlGnBu")
    # Creating confusion matrix
create_confusion_matrix(forest, X_test, y_test, figsize=(4, 3))
    


# 135 correctly classified as not mentally ill, 50 wrongly classified as not mentally ill
# 28 people wrongly classified as mentally ill, 165 correctly classified as mentally ill

# In[235]:


#Important Features
importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(forest.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.plot.bar()


# In[236]:


# Printing feature importances
print(
    pd.DataFrame(
        forest.feature_importances_, columns=["Imp"], index=X_train.columns
    ).sort_values(by="Imp", ascending=False)
)


# In[237]:


feature_names = list(X.columns)
print(feature_names)


# In[238]:


#Precision Recall Curve
from sklearn.metrics import precision_recall_curve

# getting the probabilities of our predictions
y_scores = forest.predict_proba(X_train)
y_scores = y_scores[:,1]

precision, recall, threshold = precision_recall_curve(y_train, y_scores)
def plot_precision_and_recall(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], "orange", label="precision", linewidth=5)
    plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)
    plt.xlabel("threshold", fontsize=19)
    plt.legend(loc="upper right", fontsize=19)
    plt.ylim([0, 1])

plt.figure(figsize=(14, 7))
plot_precision_and_recall(precision, recall, threshold)
plt.show()


# In[239]:


from sklearn.metrics import roc_auc_score
y_scores = forest.predict_proba(X_train)
y_scores = y_scores[:,1]
r_a_score = roc_auc_score(y_train, y_scores)
print("ROC-AUC-Score:", r_a_score)


# In[241]:


#ROC AUC Curve
from sklearn.metrics import roc_curve
# compute true positive rate and false positive rate
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, y_scores)
# plotting them against each other
def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'orange', linewidth=4)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    plt.ylabel('True Positive Rate (TPR)', fontsize=16)

plt.figure(figsize=(14, 7))
plot_roc_curve(false_positive_rate, true_positive_rate)
plt.show()


# In[242]:


from sklearn.metrics import roc_auc_score
r_a_score = roc_auc_score(y_train, y_scores)
print("ROC-AUC-Score:", r_a_score)


# In[244]:


# testing performance comparison

models_test_comp_df = pd.concat(
    [
        logistic_regression_perf.T,
        dTree_model_test_perf.T,
        forest_model_test_perf.T,
    ],
    axis=1,
)
models_test_comp_df.columns = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
   
]
print("Testing performance comparison:")
models_test_comp_df


# In[ ]:




