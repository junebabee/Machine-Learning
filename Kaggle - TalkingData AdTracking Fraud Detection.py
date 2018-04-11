
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import time
import gc


# In[22]:


from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import xgboost as xgb
import lightgbm as lgb


# In[3]:


get_ipython().run_cell_magic('time', '', "\ncolumn_types = {'click_id': 'uint32',\n                 'app': 'uint16',\n                 'channel': 'uint16',\n                 'device': 'uint16',\n                 'ip': 'uint32',\n                 'is_attributed': 'uint8',\n                 'os': 'uint16'}\n\ntrain_data = pd.read_csv('train.csv',\n                         skiprows = range(1,184903890-10000000),\n                         nrows = 10000000,\n                         dtype = column_types,\n                         usecols = ['ip','app','device','os', 'channel', 'click_time', 'is_attributed'],\n                         parse_dates = ['click_time'],\n                         infer_datetime_format = True)\ntest_data = pd.read_csv('test.csv',\n                        dtype = column_types,\n                        usecols = ['ip','app','device','os', 'channel', 'click_time', 'click_id'],\n                        parse_dates = ['click_time'],\n                        infer_datetime_format = True)\n\ntrain_size = train_data.shape[0]\n\nprint ('train size: ', train_data.shape, 'test size: ', test_data.shape)")


# In[4]:


full_data=pd.concat([train_data,test_data])


# In[5]:


del train_data, test_data
full_data['click_id'] = full_data['click_id'].fillna(0).astype('uint32')
full_data['is_attributed'] = full_data['is_attributed'].fillna(0).astype('uint8')
gc.collect()


# In[6]:


cat_var = []
num_var = ['app','channel','device','ip','os']
target_var = 'is_attributed'
id_var = 'click_id'


# In[7]:


full_data.head()


# In[8]:


OHE = OneHotEncoder(sparse = True)
start = time.time()
OHE.fit(full_data[num_var])
OHE_sparse = OHE.transform(full_data[num_var])

print ('One-hot-encoding finished in %f seconds' % (time.time()-start))


# In[10]:


OHE_sparse.shape


# In[12]:


full_data['hour'] = pd.to_datetime(full_data.click_time).dt.hour.astype('uint8')
full_data['day'] = pd.to_datetime(full_data.click_time).dt.day.astype('uint8')
full_data['dow']  = pd.to_datetime(full_data.click_time).dt.dayofweek.astype('uint8')

dt_vars = ['hour','day', 'dow']
gc.collect()


# In[16]:


tmp = full_data[['ip','day','hour','channel']].groupby(by=['ip','day','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'qty'})
tmp['qty'] = tmp['qty'].astype('uint16')
full_data = full_data.merge(tmp, on=['ip','day','hour'], how='left')
del tmp
gc.collect()

tmp = full_data[['ip', 'app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
tmp['ip_app_count'] = tmp['ip_app_count'].astype('uint16')
full_data = full_data.merge(tmp, on=['ip','app'], how='left')
del tmp
gc.collect()

tmp = full_data[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
tmp['ip_app_os_count'] = tmp['ip_app_os_count'].astype('uint16')
full_data = full_data.merge(tmp, on=['ip','app', 'os'], how='left')
del tmp
gc.collect()

interact_vars = ['qty', 'ip_app_count', 'ip_app_os_count']


# In[21]:


full_data.head()


# In[23]:


full_vars = num_var + dt_vars + interact_vars
print ('Training variables are: ',full_vars)
train_x = full_data[full_vars][:train_size].values
train_y = full_data[target_var][:train_size].values
test_x = full_data[full_vars][train_size:].values
ids = full_data[id_var][train_size:].values
gc.collect()
print ('Train data size: ',train_x.shape, 'Test data size: ',test_x.shape)


# In[27]:


lgb_params = {
    'boosting_type' : 'gbdt', # gbdt:traditional Gradient Boosting Decision Tree; rf: Random Forest
    'objective' : 'binary',
    'metric' : 'auc',
    'learning_rate' : 0.15,
    'num_leaves' : 15,
    'max_depth' : 4,
    'min_child_samples' : 100,
    # minimal number of data in one leaf. Can be used to deal with over-fitting
    'max_bin' : 100,
    # max number of bins that feature values will be bucketed in. 
    # Small number of bins may reduce training accuracy but may increase general power (deal with over-fitting)
    # LightGBM will auto compress memory according max_bin. For example, 
    # LightGBM will use uint8_t for feature value if max_bin=255
    'subsample' : .7,
    # like feature_fraction, but this will randomly select part of data without resampling
    # can be used to speed up training
    # can be used to deal with over-fitting
    # Note: To enable bagging, bagging_freq should be set to a non zero value as well
    'subsample_freq' : 1,
    # frequency for bagging, 0 means disable bagging. k means will perform bagging at every k iteration
    # Note: to enable bagging, bagging_fraction should be set as well
    'colsample_bytree' : 0.7,
    # LightGBM will randomly select part of features on each iteration if feature_fraction smaller than 1.0. 
    # For example, if set to 0.8, will select 80% features before training each tree
    # can be used to speed up training
    # can be used to deal with over-fitting
    'min_child_weight' : 0, 
    # minimal sum hessian in one leaf. Like min_data_in_leaf, it can be used to deal with over-fitting
    'scale_pos_weight' : 99 # default=1.0, type=double, weight of positive class in binary classification task
}

cv_results = lgb.cv(lgb_params,
                    lgb.Dataset(train_x,train_y),
                    num_boost_round=200,
                    nfold=4,
                    early_stopping_rounds=50,
                    feval=None, # custom evaluation function
                    stratified=True, # whether to perform 分层抽样
                    shuffle=True, # whether to shuffle before splitting data
                    verbose_eval=10 # whether to display the progress
                   ) 
gc.collect()

cv_results = pd.DataFrame(cv_results)
best_lgb_iteration = len(cv_results)
best_lgb_score = cv_results['auc-mean'].max()
print('best_score:', best_lgb_score, 'best_iteration:', best_lgb_iteration)


# In[29]:


model = lgb.train(lgb_params,
                  lgb.Dataset(train_x, train_y),
                  num_boost_round=best_lgb_iteration
                  )

preds = model.predict(test_x)

sub_df = pd.DataFrame({'click_id': ids, 'is_attributed': preds})
sub_df.to_csv("lgb_starter.csv", index=False)


# In[30]:


sub_df.shape

