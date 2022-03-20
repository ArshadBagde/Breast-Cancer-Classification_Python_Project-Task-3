#!/usr/bin/env python
# coding: utf-8

# # Project:- Breast Cancer Classification Python Project
# Organization:- Infopillar Solution 

# #  Data Science Internship
# Author:- Arshad R. Bagde

# In[ ]:





# In[1]:


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 100)

import seaborn as sns
import matplotlib.pyplot as plt


# ## Exploratory Data Analysis (EDA)

# ### Introductory Details

# In[2]:


# read the csv file
df = pd.read_csv('breast-cancer.csv')


# In[3]:


# print the number of rows and the number of columns
print(f'The dataset has {df.shape[0]} rows and {df.shape[1]} columns.')


# In[4]:


# exhibit the first ten observations
df.head()


# In[5]:


# exhibit the bottom ten observations
df.tail()


# In[6]:


# show the data types
df.dtypes


# ### Data Cleaning

# #### Missing Values

# In[7]:


# count the number of missing values
df.isna().sum()


# In[8]:


# count the number of duplicate rows
df.duplicated().sum()


# In[9]:


# drop the id column
df.drop(columns='id', inplace=True)


# ### Descriptive Statistics

# #### Target Variable

# In[10]:


labels = df['diagnosis'].value_counts().index.tolist()
values = np.round((df['diagnosis'].value_counts(normalize=True)*100), 1).to_list()

# creating the bar plot
plt.bar(labels, 
        values, 
        color = ['#e9731f', '#0470b4'],
        width = 0.7)

plt.xlabel("Cancer Type")
plt.ylabel("No. of Diagnosis (%)")
plt.title("Target Variable Count")

# add the value on each bar
for i in range(len(values)):
        plt.text(i, values[i]//2, values[i], ha='center', color='white')

plt.show()

print(f'The benign cancers represent {values[0]}% of the cases')
print(f'The malignant cancers represent {values[1]}% of the cases')
print()


# #### Independent Variables

# In[11]:


# exhibit some basic descriptive statistics
df.describe().round(4)


# In[12]:


# split the dataset into 3 dataframes
df_mean = df.iloc[:, 1:11]
df_se = df.iloc[:, 11:21]
df_worst = df.iloc[:, 21:]

# extract the target variable
y = df['diagnosis']


# #### Data Distribution

# In[13]:


def min_max_scaler(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform features by scaling each feature to a range from 0 to 1.
    """
    minimum = df.min()
    
    df = (df - minimum) / (df.max() - minimum)
    
    return df

def df_melter(y: pd.Series, df: pd.DataFrame) -> pd.DataFrame:
    """
    Modify the selected columns of a DataFrame from wide to long format.
    """
    df = pd.concat([y, df], axis=1)

    return pd.melt(df, id_vars='diagnosis', var_name='features', value_name='value')

def subplot_generator(df: pd.DataFrame, plot: str, y=None) -> None:
    """
    Prepare the main plot to be filled out with subplots.
    """
    df = min_max_scaler(df)

    fig, axes = plt.subplots(2, 5, figsize=(16, 12), sharey=True)
    fig.tight_layout(pad=3.0)
    axes = axes.flatten()

    plot_feature(df, plot, axes, y)
    
def plot_feature(df: pd.DataFrame, plot: str, axes, y=None) -> None:
    """
    Create the selected plot for each future in the DataFrame.
    """
    for i, feature in enumerate(df.columns):
        if plot == 'histogram':
            histoplot(df, feature, i, axes)
        elif plot == 'violinplot':
            violinplot(df, y, feature, i, axes)
        elif plot == 'swarmplot':
            swarmplot(df, y, feature, i, axes)
        elif plot == 'boxplot':
            boxplot(df, y, feature, i, axes)


def histoplot(df: pd.DataFrame, feature: str, i: int, axes) -> None:
    """
    Prepare the several histograms to enter the main plot.
    """

    bin_count = int(np.ceil(np.log2(len(df))) + 1)

    sns.histplot(ax=axes[i],
                data=df,
                x=feature,
                bins=bin_count,
                kde=True,
                line_kws={'lw': 3})

    plot_polishing(axes=axes, i=i, plot='histogram')

def violinplot(df: pd.DataFrame, y: pd.Series, feature: str, i: int, axes) -> None:
    """
    Prepare the several violinplots to enter the main plot.
    """

    df = df_melter(y, df)

    sns.violinplot(ax=axes[i],
                data=df[df['features'] == feature],
                x='features', 
                y='value', 
                hue='diagnosis',
                split=True)

    plot_polishing(axes=axes, i=i, plot='violinplot')

def swarmplot(df: pd.DataFrame, y: pd.Series, feature: str, i: int, axes) -> None:
    """
    Prepare the several swarmplots to enter the main plot.
    """
    
    df = df_melter(y, df)

    sns.swarmplot(ax=axes[i],
                data=df[df['features'] == feature],
                x='features', 
                y='value', 
                hue='diagnosis',
                size=3)
    
    plot_polishing(axes=axes, i=i, plot='swarmplot')

def boxplot(df: pd.DataFrame, y: pd.Series, feature: str, i: int, axes) -> None:
    """
    Prepare the several swarmplots to enter the main plot.
    """

    sns.boxplot(ax=axes[i],
                data=df,
                x=feature,
                orient='v',
                width=0.3,
                flierprops={'marker' : 'x',
                            'markeredgecolor' : 'red', 
                            'markersize' : 6})
    
    plot_polishing(axes=axes, i=i, plot='histogram')


def plot_polishing(axes, i: int, plot: str) -> None:
    """
    Eliminate the clutter from the plots. 
    """
    axes[i].spines['top'].set_visible(False)
    axes[i].spines['right'].set_visible(False)
    axes[i].spines['left'].set_visible(False)
    axes[i].get_yaxis().set_visible(False)

    if plot in {'boxplot', 'histogram'}:
        axes[i].set(ylabel='')
        axes[i].set_xticks([])
    else:
        axes[i].get_legend().remove()
        axes[i].set(xlabel='', ylabel='')


# ##### Histograms

# In[14]:


# ceiling the result of the logarithm ensures the result to be an integer
bin_count = int(np.ceil(np.log2(len(df))) + 1)

print(f'Number of bins according to Sturge\'s rule: {bin_count}')


# In[15]:


# plot the histograms
subplot_generator(df=df_mean, plot='histogram')


# In[16]:


# plot the histograms
subplot_generator(df=df_se, plot='histogram')


# In[17]:


# plot the histograms
subplot_generator(df=df_worst, plot='histogram')


# ##### Violin Plot

# In[18]:


# plot the violinplots
subplot_generator(df=df_mean, y=y, plot='violinplot')


# In[19]:


# plot the violinplots
subplot_generator(df=df_se, y=y, plot='violinplot')


# In[20]:


# plot the violinplots
subplot_generator(df=df_worst, y=y, plot='violinplot')


# ##### Swarm Plot

# In[21]:


# plot the swarmplots
subplot_generator(df=df_mean, y=y, plot='swarmplot')


# In[22]:


# plot the swarmplots
subplot_generator(df=df_se, y=y, plot='swarmplot')


# In[23]:


# plot the swarmplots
subplot_generator(df=df_worst, y=y, plot='swarmplot')


# ### Outliers

# In[24]:


# plot the boxplot
subplot_generator(df=df_mean, plot='boxplot')


# In[25]:


# plot the boxplot
subplot_generator(df=df_se, plot='boxplot')


# In[26]:


# plot the boxplot
subplot_generator(df=df_worst, plot='boxplot')


# In[27]:


from sklearn.neighbors import LocalOutlierFactor

# find the outliers
y_pred = LocalOutlierFactor().fit_predict(df.drop(["diagnosis"], axis=1))

outlier_count = abs(sum(y_pred[y_pred < 1]))

print(f'The vanilla Local Outlier Factor identified {outlier_count} outliers ({round(outlier_count/len(df), 2)}%)' )


# ### Correlation

# In[28]:


# encoding the target variable  
df['diagnosis'] = df['diagnosis'].map({'B' : 0, 'M' : 1})

# convert it into a numeric variable
df['diagnosis'] = pd.to_numeric(df['diagnosis'])


# In[29]:


plt.figure(figsize=(30, 24))

# create the correlation matrix
corr_mat = df.corr(method='spearman')

# remove the upper diagonal
mask = np.zeros(corr_mat.shape, dtype=bool)
mask[np.triu_indices(len(mask))] = True

# set the plot dimension
fig = plt.figure(1, figsize=(14, 10))

ax = sns.heatmap(corr_mat, vmin=-1, vmax=1, center=0,
                 cmap=sns.diverging_palette(20, 220, n=100),
                 square=True, annot=True, mask=mask)

# modify the X and Y labels appearence
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
ax.set_yticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

# plot the graph
plt.show()


# In[30]:


high_corr_feat_list = list(corr_mat[(abs(corr_mat['diagnosis']) >= 0.7) & (corr_mat.columns != 'diagnosis')].index)
high_corr_feat_list


# |                     |        Feature 1       |     Feature 2     |        Feature 3        |      Feature 4     |     Feature 5    |
# |---------------------|:----------------------:|:-----------------:|:-----------------------:|:------------------:|:----------------:|
# | *radius_mean*       | *area_worst*           | *perimeter_worst* | *radius_worst*          | *area_mean*        | *perimeter_mean* |
# | *concavity_mean*    | *concave_points_worst* | *concavity_worst* | *concavity_points_mean* | *compactness_mean* |                  |
# | *radius_se*         | *perimeter_se*         | *area_se*         |                         |                    |                  |
# | *compactness_mean*  | *compactness_worst*    |                   |                         |                    |                  |
# | *compactness_worst* | *concavity_worst*      |                   |                         |                    |                  |
# | *texture_mean*      | *texture_worst*        |                   |                         |                    |                  |

# ## Model Selection

# In[31]:


from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from skopt.sampler import Grid

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn import FunctionSampler

from sklearn.metrics import recall_score


# In[32]:


def nested_cv(X: pd.DataFrame, y: pd.Series, cv_outer: StratifiedKFold, opt_search: BayesSearchCV, validation_result: bool) -> None:
    """
    Run the nested cross validation.
    """
    outer_results, inner_results = outer_loop(X=X, y=y, cv_outer=cv_outer, opt_search=opt_search, validation_result=validation_result)

    # print the CV overall results
    print(f'Recall | Validation Mean: {round(np.mean(inner_results), 3)}, Validation Std: {round(np.std(inner_results), 3)}')
    print(f'Recall | Test Mean: {round(np.mean(outer_results), 3)}, Test Std: {round(np.std(outer_results), 3)}')
    

def outer_loop(X: pd.DataFrame, y: pd.Series, cv_outer: StratifiedKFold, opt_search: BayesSearchCV, validation_result: bool) -> list:
    """
    Perform the outer loop split and per each fold, its inner loop.
    """
    outer_results, inner_results = [], []
    
    for i, (train_index, test_index) in enumerate(cv_outer.split(X, y), start=1):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # start the Bayes search
        _ = opt_search.fit(X_train, y_train)

        # save the best model
        best_model = opt_search.best_estimator_

        # predict on the test set
        y_pred = best_model.predict(X_test)

        # calculate the recall on test set
        recall = recall_score(y_test, y_pred)
        
        # append the recall results
        outer_results.append(recall)
        inner_results.append(opt_search.best_score_)
        
        print_validation_results(i=i, opt_search=opt_search, recall=recall, validation_result=validation_result)
    
    return outer_results, inner_results


def print_validation_results(i: int, opt_search: BayesSearchCV, recall: float, validation_result: bool) -> None:
    """
    Print the validation results per each fold.
    """
    if validation_result:
        print(f'Fold {i}')
        print(f'Recall | Validation: {round(opt_search.best_score_, 3)}\tTest: {round(recall, 3)}')
        print('\n')
        print(f'Best Hyperparameter Combination:\n{opt_search.best_params_}')
        print('\n')


# In[33]:


# set the inner and outer CV
cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
cv_inner = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)


# In[34]:


# set the dict with params to
# be passed onto the optimizer
optimizer_dict = {
        'n_initial_points' : 10,
        'initial_point_generator' : Grid(border="include")
        }


# ### Data Preprocessing

# In[35]:


# split the columns into features and targets
X = df[df.columns.drop(['diagnosis'])]
y = df['diagnosis']


# ### Baseline Model

# In[36]:


# create the pipeline
pipeline = Pipeline([
    ('clf', None)
])


# In[37]:


# set the parameter search space for classifier 1
gnb_search = {
    'clf': Categorical([GaussianNB()]),
    'clf__var_smoothing': Real(1e-9, 2)
}

# set the parameter search space for classifier 2
svc_search = {
    'clf': Categorical([SVC()]),
    'clf__C': Real(1e-6, 1e+6, prior='log-uniform'),
    'clf__gamma': Real(1e-6, 1e+1, prior='log-uniform'),
    'clf__degree': Integer(1, 3),
    'clf__kernel': Categorical(['linear', 'poly', 'rbf'])
}

# set the parameter search space for classifier 3
log_search = {
    'clf': Categorical([LogisticRegression()]),
    'clf__C': Real(1e-5, 10),
    'clf__penalty': Categorical(['l2']),
    'clf__class_weight': Categorical([None, 'balanced']),
    'clf__solver': Categorical(['lbfgs', 'liblinear']),
    'clf__max_iter': [1000]
}

# set the parameter search space for classifier 4
rf_search = {
    'clf': Categorical([RandomForestClassifier()]),
    'clf__n_estimators': Integer(10, 200),
    'clf__criterion': Categorical(['gini', 'entropy']),
    'clf__min_samples_split': Integer(2, 200),
    'clf__min_samples_leaf': Integer(1, 200),
    'clf__min_impurity_decrease': Real(0, 1),
    'clf__max_features': Integer(1, 15)
}

# set the parameter search space for classifier 1
knn_search = {
    'clf': Categorical([KNeighborsClassifier()]),
    'clf__n_neighbors': Integer(2, 20, prior='log-uniform'),
    'clf__weights': Categorical(['uniform', 'distance']),
    'clf__leaf_size': Integer(30, 100),
    'clf__p': Integer(1, 2),
    'clf__algorithm': Categorical(['ball_tree', 'kd_tree', 'brute'])
}


# In[38]:


# create a list of models
model_list = [
    gnb_search,
    svc_search,
    log_search,
    rf_search,
    knn_search
]


# In[39]:


# set the number of searches to perform
search_num = 10

# create the search space by joining
# all of the classifiers that need to be optimized
search_space_list = [
    (log_search, search_num), 
    (svc_search, search_num), 
    (rf_search, search_num),
    (gnb_search, search_num),
    (knn_search, search_num)
]


# In[40]:


# define the Bayesian search
opt_search = BayesSearchCV(
    estimator=pipeline,
    search_spaces=search_space_list,
    optimizer_kwargs=optimizer_dict,
    scoring='recall',
    cv=cv_inner,
    refit=True,
    return_train_score=True,
    random_state=42)


# In[41]:


X, y


# In[42]:


# %%time
# # enumerate splits
# nested_cv(X, y, cv_outer, opt_search, True)


# ### Baseline Model + Scaling

# In[43]:


# create the pipeline with the scaler
pipeline = Pipeline([
    ('scaler', MinMaxScaler()), 
    ('clf', None)
])


# In[44]:


# define the Bayesian search
opt_search = BayesSearchCV(
    estimator=pipeline,
    search_spaces=search_space_list,
    optimizer_kwargs=optimizer_dict,
    scoring='recall',
    cv=cv_inner,
    refit=True,
    return_train_score=True,
    random_state=42)


# In[45]:


get_ipython().run_cell_magic('time', '', '# enumerate splits\nnested_cv(X, y, cv_outer, opt_search, False)')


# ### Baseline Model + Scaling + Outlier Detection

# In[46]:


def lof(X, y):
    """Find the outliers above the 1st percentile and remove them from both X and y."""
    model = LocalOutlierFactor()
    model.fit(X)
    # extract 
    lof_score = model.negative_outlier_factor_
    # find the 1st percentile
    percentile = np.quantile(lof_score, 0.01)

    return X[lof_score > percentile, :], y[lof_score > percentile]


# In[47]:


# create the pipeline
pipeline = Pipeline([
    ('outlier_detector', FunctionSampler(func=lof)),
    ('scaler', MinMaxScaler()), 
    ('clf', None)
])


# In[48]:


# define the Bayesian search
opt_search = BayesSearchCV(
    estimator=pipeline,
    search_spaces=search_space_list,
    optimizer_kwargs=optimizer_dict,
    scoring='recall',
    cv=cv_inner,
    refit=True,
    return_train_score=True,
    random_state=42)


# In[49]:


get_ipython().run_cell_magic('time', '', '# enumerate splits\nnested_cv(X, y, cv_outer, opt_search, False)')


# ### Baseline Model + Scaling + SMOTE

# In[50]:


# create the pipeline
pipeline = Pipeline([
    ('smote', SMOTE(sampling_strategy='minority', random_state=42)),
    ('scaler', MinMaxScaler()), 
    ('clf', None)
])

# add SMOTE parameters
for d in model_list:
    d['smote__k_neighbors'] = Integer(2, 15)


# In[51]:


# define the Bayesian search
opt_search = BayesSearchCV(
    estimator=pipeline,
    search_spaces=search_space_list,
    optimizer_kwargs=optimizer_dict,
    scoring='recall',
    cv=cv_inner,
    refit=True,
    return_train_score=True,
    random_state=42)


# In[52]:


get_ipython().run_cell_magic('time', '', '# enumerate splits\nnested_cv(X, y, cv_outer, opt_search, False)')

