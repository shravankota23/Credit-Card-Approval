#!/usr/bin/env python
# coding: utf-8

# In[45]:


# Import required packages
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import classification_report as cr
import pickle


# In[46]:


# load dataset
cc_apps=pd.read_csv("crx.data",delimiter=",",names=["Gender","Age","Debt","Married","BankCustomer","Educational","Ethinicity","YearsEmployed","PriorDefault","Employed","CreditScore","DriversLicense","Citizen","ZipCode","Income","ApprovalStatus"])


# In[47]:


cc_apps.tail(17)


# In[48]:


cc_apps.describe()


# In[49]:


cc_apps.info()


# In[50]:


cc_apps["Gender"].value_counts()


# In[51]:


cc_apps.replace("?",np.nan,inplace=True)


# In[52]:


cc_apps["Age"]=cc_apps["Age"].astype("float")


# In[53]:


cc_apps.info()


# In[54]:


cc_apps.isna().sum()


# In[55]:


obj=cc_apps.columns[cc_apps.dtypes=="object"]


# In[56]:


obj


# In[57]:


num=cc_apps.columns[cc_apps.dtypes!="object"]


# In[58]:


num


# In[59]:


obj_imp=SimpleImputer(strategy="most_frequent")
num_imp=SimpleImputer()


# In[60]:


cc_apps[obj]=obj_imp.fit_transform(cc_apps[obj])
cc_apps[num]=num_imp.fit_transform(cc_apps[num])


# In[61]:


cc_apps["ApprovalStatus"].replace({"+":1,"-":0},inplace=True)


# In[62]:


cc_apps.isna().sum()


# In[63]:


le=LabelEncoder()

for col in cc_apps.columns:
    if cc_apps[col].dtypes=="object":
        cc_apps[col]=le.fit_transform(cc_apps[col])
        


# In[64]:


cc_apps.drop(["ZipCode","DriversLicense"],axis=1,inplace=True)


# In[65]:


X=cc_apps.drop("ApprovalStatus",axis=1)
y=cc_apps["ApprovalStatus"]


# In[66]:


x_train,x_test,y_train,y_test=tts(X,y,test_size=0.2,random_state=123,stratify=y)


# In[67]:


scaler=MinMaxScaler(feature_range=(0,1))
scaled_x_train=scaler.fit_transform(x_train)
scaled_x_test=scaler.transform(x_test)


# In[68]:


lr=LogisticRegression()


# In[69]:


lr.fit(scaled_x_train,y_train)


# In[70]:


y_pred=lr.predict(scaled_x_test)


# In[71]:


lr.score(x_test,y_test)


# In[72]:


cm(y_test,y_pred)


# In[73]:


print(cr(y_test,y_pred))


# <b>Overall Accuracy of the model is 0.86<b><br>
# <b>We can see that precision as well as recal and f1 score are near to 0.9 so we can say that our model is good fit<b>

# Pickle is used to deploy models further

# In[74]:


with open('ccapproval.pkl', 'wb') as f:
    pickle.dump(lr, f)


# In[75]:


# import pickle
# from sklearn.metrics import confusion_matrix as cm
# from sklearn.metrics import classification_report as cr
# with open('ccapproval.pkl', 'rb') as f:
#     model1 = pickle.load(f)
# type(model1)
# y_predd=model1.predict(x_test)
# print(cm(y_test,y_predd))
# print(cr(y_test,y_predd))

