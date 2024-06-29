#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression

# In[63]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import recall_score,f1_score,precision_score


# In[2]:


df = pd.read_csv("Dataset.csv")
df.head()


# In[3]:


df.info()


# # Null Value

# In[4]:


# Check Null value
df.isnull().sum()/len(df) * 100


# # Outlier Analysis

# In[5]:


for column in df.columns:
    sns.boxplot(column, data=df)
    plt.show()


# In[6]:


# There are outliers in AccountWeeks, DataUasage, CustServCalls,DayMins, DayCalls, MonthlyCharge, OverageFee, and RoamMins. 


# In[7]:


# # Remove Outlier Analysis
# # Flooring and Capping

# for column in df.columns:
#     q1 = df[column].quantile(0.25)
#     q3 = df[column].quantile(0.75)
#     iqr = q3 - q1
#     ll = q1 - (1.5 * iqr)
#     ul = q3 + (1.5 * iqr)
#     #print(column,ll,ul)
#     for index in df[column].index:
#         #print(df.loc[index,column])
#         if df.loc[index,column]>ul:
#             df.loc[index,column]=ul
#         if df.loc[index,column]<ll:
#             df.loc[index,column]=ll


# In[8]:


for column in df.columns:
    sns.boxplot(column, data=df)
    plt.show()


# # EDA 

# In[9]:


df.head()


# In[10]:


df.info()


# In[11]:


df.columns


# In[12]:


for col in df.columns:
    print(col)
    print(df[col].unique())


# In[13]:


cont_cols = ['AccountWeeks','OverageFee','RoamMins']
cat_cols = list(set(df.columns) - set(cont_cols))
print("Continuous variables: ", cont_cols)
print("Categorical variables: ", cat_cols)


# <b> Univariate Anlaysis  </b>

# In[14]:


# Continuous Data - Histogram or KDE


# In[15]:


for col in cont_cols:
    sns.kdeplot(col,data=df)
    plt.show()


# In[16]:


for col in cont_cols:
    print(col, ": " , df[col].skew())


# In[17]:


# AccountWeeks, OverageFee and RoamMins are closer to uniform distribution - closer to 0


# In[18]:


# Categorical Data - Bar graphs


# In[19]:


#plt.figure(figsize=(5,5))
for col in cat_cols:
    sns.countplot(col,data=df)
    plt.show()


# In[20]:


for col in df.columns:
    print(col)
    print(df[col].value_counts(normalize=True).round(2))


# In[21]:


for col in cont_cols:
    sns.boxplot(x='Churn',y=col,data=df)
    plt.show()


# In[22]:


# AccountWeeks, OverageFee and RoamMins has no impact on defaulter


# In[23]:


# Target vs Categorical data

# Categorical vs Categorical data

# Stacked bar plot


# In[24]:


for col in cat_cols:
    pd.crosstab(df[col],df['Churn']).plot(kind='bar')
    plt.show()


# In[25]:


# Contract Renewal 1 an Data plan 0 are having more churn counts


# # Scaling

# In[26]:


# Scaling - Continuous data


# In[27]:


# from sklearn.preprocessing import StandardScaler


# In[28]:


ss = StandardScaler()
x_con_scaled = pd.DataFrame(ss.fit_transform(df[cont_cols]), columns=cont_cols,index=df.index)
x_con_scaled.head()


# In[29]:


ss = StandardScaler()
x_cat_scaled = pd.DataFrame(ss.fit_transform(df[cat_cols]), columns=cat_cols, index=df.index)
x_cat_scaled.head()


# # Encoding

# In[30]:


# Categorical data - Numerical Data

# One hot encoding


# In[31]:


x_cal_enc = pd.get_dummies(df[cat_cols],drop_first=True) # function to execute one hot encoding 
x_cal_enc.head()


# # Marge Cat and Con Data

# In[32]:


x_final = pd.concat([x_con_scaled,x_cal_enc],axis=1)
x_final


# # Train Test Split

# In[33]:


#from sklearn.model_selection import train_test_split


# In[34]:


y = df['Churn']


# In[35]:


# Training = 80%, Testing = 20% ( Random Selection of train_test_split)


# In[36]:


x_train,x_test, y_train,y_test = train_test_split(x_final,y,test_size = 0.2, random_state=10)


# # Logistic Regression Model

# In[37]:


# from sklearn.linear_model import LogisticRegression


# In[38]:


log_reg = LogisticRegression(penalty='l2',C=1.0, class_weight='balanced',fit_intercept=True)
log_reg.fit(x_train,y_train)


# In[39]:


print('Intercept (b0): ', log_reg.intercept_)
print('Coerricients (b1): ', log_reg.coef_[0])


# # Performance Metrics

# In[40]:


y_test_pred = log_reg.predict(x_test)


# In[41]:


#from sklearn.metrics import confusion_matrix,classification_report


# In[42]:


sns.heatmap(confusion_matrix(y_test,y_test_pred),annot=True,fmt="0.0f")


# In[43]:


print(classification_report(y_test,y_test_pred))


# # ROC Curve

# In[ ]:


#from sklearn.metrics import roc_curve, roc_auc_score


# In[52]:


y_test_pred_prob = log_reg.predict_proba(x_test)[:,1]
fpr,tpr,thresh = roc_curve(y_test,y_test_pred_prob,drop_intermediate=True) #Y_actual , Y_predict_Probability
roc_df = pd.DataFrame({'TPR':tpr,'FPR':fpr,'Threshold':thresh})
roc_df


# In[53]:


plt.title('ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.plot(roc_df['FPR'],roc_df['TPR'])
plt.plot([[0,0],[1,1]],color='red')
plt.show()


# In[54]:


print('ROC_AUC:',roc_auc_score(y_test,y_test_pred_prob))


# # Train & Test Score

# In[44]:


y_train_pred = log_reg.predict(x_train)
y_test_pred = log_reg.predict(x_test)


# In[45]:


print('Train Confusion Matrix: ')
sns.heatmap(confusion_matrix(y_train,y_train_pred), annot=True, fmt='0.0f')
plt.show()
print('Test Confusion Matrix: ')
sns.heatmap(confusion_matrix(y_test,y_test_pred), annot=True, fmt='0.0f')
plt.show()


# In[46]:


print('Train Classification Report: ')
print(classification_report(y_train,y_train_pred))
print('-'*70)
print('Test Classification Report: ')
print(classification_report(y_test,y_test_pred))


# In[55]:


y_train_pred_prod=log_reg.predict_proba(x_train)[:,1]
y_test_pred_prod=log_reg.predict_proba(x_test)[:,1]


# In[56]:


print('Train ROC: ',roc_auc_score(y_train,y_train_pred_prod))
print('Test ROC: ',roc_auc_score(y_test,y_test_pred_prod))


# # Cross Validation Score

# In[47]:


# from sklearn.model_selection import cross_val_score


# In[48]:


scores = cross_val_score(log_reg,x_train,y_train,scoring='recall',cv=5)


# In[49]:


print('Score: ',scores)
print('Avg Score: ',np.mean(scores))
print('Std Score:', np.std(scores))


# In[50]:


# Model is having constant performance in all the splits


# # Finding model threshold by ROC

# In[58]:


roc_df


# In[60]:


roc_df['Difference'] = roc_df['TPR'] - roc_df['FPR']
roc_df[roc_df['Difference'] == max(roc_df['Difference'])]


# # Choosing threshold by a performance metric

# In[62]:


# from sklearn.metrics import recall_score,f1_score,precision_score


# In[64]:


thresholds = np.arange(0,1.1,0.1)
recall_list = []
f1_list = []
precision_list = []
for th in thresholds:
    y_pred_class = [0 if pval < th else 1 for pval in y_test_pred_prob]
    recall_list.append(recall_score(y_test,y_pred_class))
    f1_list.append(f1_score(y_test,y_pred_class))
    precision_list.append(precision_score(y_test,y_pred_class))


# In[65]:


plt.plot(thresholds,recall_list,color='red')
plt.plot(thresholds,precision_list,color='blue')
plt.plot(thresholds,f1_list,color='black')


# In[ ]:




