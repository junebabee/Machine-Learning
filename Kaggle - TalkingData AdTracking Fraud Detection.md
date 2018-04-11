

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import time
import gc
```


```python
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import xgboost as xgb
import lightgbm as lgb
```


```python
%%time

column_types = {'click_id': 'uint32',
                 'app': 'uint16',
                 'channel': 'uint16',
                 'device': 'uint16',
                 'ip': 'uint32',
                 'is_attributed': 'uint8',
                 'os': 'uint16'}

train_data = pd.read_csv('train.csv',
                         skiprows = range(1,184903890-10000000),
                         nrows = 10000000,
                         dtype = column_types,
                         usecols = ['ip','app','device','os', 'channel', 'click_time', 'is_attributed'],
                         parse_dates = ['click_time'],
                         infer_datetime_format = True)
test_data = pd.read_csv('test.csv',
                        dtype = column_types,
                        usecols = ['ip','app','device','os', 'channel', 'click_time', 'click_id'],
                        parse_dates = ['click_time'],
                        infer_datetime_format = True)

train_size = train_data.shape[0]

print ('train size: ', train_data.shape, 'test size: ', test_data.shape)
```

    train size:  (10000000, 7) test size:  (18790469, 7)
    CPU times: user 2min 20s, sys: 43.1 s, total: 3min 4s
    Wall time: 3min 20s



```python
full_data=pd.concat([train_data,test_data])
```


```python
del train_data, test_data
full_data['click_id'] = full_data['click_id'].fillna(0).astype('uint32')
full_data['is_attributed'] = full_data['is_attributed'].fillna(0).astype('uint8')
gc.collect()
```




    25




```python
cat_var = []
num_var = ['app','channel','device','ip','os']
target_var = 'is_attributed'
id_var = 'click_id'
```


```python
full_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>app</th>
      <th>channel</th>
      <th>click_id</th>
      <th>click_time</th>
      <th>device</th>
      <th>ip</th>
      <th>is_attributed</th>
      <th>os</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18</td>
      <td>439</td>
      <td>0</td>
      <td>2017-11-09 12:58:40</td>
      <td>1</td>
      <td>159091</td>
      <td>0</td>
      <td>13</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>205</td>
      <td>0</td>
      <td>2017-11-09 12:58:40</td>
      <td>1</td>
      <td>214897</td>
      <td>0</td>
      <td>13</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>134</td>
      <td>0</td>
      <td>2017-11-09 12:58:40</td>
      <td>1</td>
      <td>29034</td>
      <td>0</td>
      <td>19</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>489</td>
      <td>0</td>
      <td>2017-11-09 12:58:40</td>
      <td>1</td>
      <td>40654</td>
      <td>0</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15</td>
      <td>278</td>
      <td>0</td>
      <td>2017-11-09 12:58:40</td>
      <td>1</td>
      <td>123623</td>
      <td>0</td>
      <td>53</td>
    </tr>
  </tbody>
</table>
</div>




```python
OHE = OneHotEncoder(sparse = True)
start = time.time()
OHE.fit(full_data[num_var])
OHE_sparse = OHE.transform(full_data[num_var])

print ('One-hot-encoding finished in %f seconds' % (time.time()-start))
```

    One-hot-encoding finished in 104.741543 seconds



```python
OHE_sparse.shape
```




    (28790469, 141783)




```python
full_data['hour'] = pd.to_datetime(full_data.click_time).dt.hour.astype('uint8')
full_data['day'] = pd.to_datetime(full_data.click_time).dt.day.astype('uint8')
full_data['dow']  = pd.to_datetime(full_data.click_time).dt.dayofweek.astype('uint8')

dt_vars = ['hour','day', 'dow']
gc.collect()
```




    4813




```python
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
```


```python
full_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>app</th>
      <th>channel</th>
      <th>click_id</th>
      <th>click_time</th>
      <th>device</th>
      <th>ip</th>
      <th>is_attributed</th>
      <th>os</th>
      <th>hour</th>
      <th>day</th>
      <th>dow</th>
      <th>ip_app_count</th>
      <th>ip_app_os_count</th>
      <th>qty</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18</td>
      <td>439</td>
      <td>0</td>
      <td>2017-11-09 12:58:40</td>
      <td>1</td>
      <td>159091</td>
      <td>0</td>
      <td>13</td>
      <td>12</td>
      <td>9</td>
      <td>3</td>
      <td>5</td>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>205</td>
      <td>0</td>
      <td>2017-11-09 12:58:40</td>
      <td>1</td>
      <td>214897</td>
      <td>0</td>
      <td>13</td>
      <td>12</td>
      <td>9</td>
      <td>3</td>
      <td>10</td>
      <td>3</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>134</td>
      <td>0</td>
      <td>2017-11-09 12:58:40</td>
      <td>1</td>
      <td>29034</td>
      <td>0</td>
      <td>19</td>
      <td>12</td>
      <td>9</td>
      <td>3</td>
      <td>77</td>
      <td>36</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>489</td>
      <td>0</td>
      <td>2017-11-09 12:58:40</td>
      <td>1</td>
      <td>40654</td>
      <td>0</td>
      <td>10</td>
      <td>12</td>
      <td>9</td>
      <td>3</td>
      <td>478</td>
      <td>28</td>
      <td>13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15</td>
      <td>278</td>
      <td>0</td>
      <td>2017-11-09 12:58:40</td>
      <td>1</td>
      <td>123623</td>
      <td>0</td>
      <td>53</td>
      <td>12</td>
      <td>9</td>
      <td>3</td>
      <td>259</td>
      <td>5</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
</div>




```python
full_vars = num_var + dt_vars + interact_vars
print ('Training variables are: ',full_vars)
train_x = full_data[full_vars][:train_size].values
train_y = full_data[target_var][:train_size].values
test_x = full_data[full_vars][train_size:].values
ids = full_data[id_var][train_size:].values
gc.collect()
print ('Train data size: ',train_x.shape, 'Test data size: ',test_x.shape)
```

    Training variables are:  ['app', 'channel', 'device', 'ip', 'os', 'hour', 'day', 'dow', 'qty', 'ip_app_count', 'ip_app_os_count']
    Train data size:  (10000000, 11) Test data size:  (18790469, 11)



```python
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
```

    [10]	cv_agg's auc: 0.955194 + 0.00359067
    [20]	cv_agg's auc: 0.966561 + 0.000919665
    [30]	cv_agg's auc: 0.969497 + 0.000816327
    [40]	cv_agg's auc: 0.972193 + 0.000701245
    [50]	cv_agg's auc: 0.974471 + 0.000601477
    [60]	cv_agg's auc: 0.975722 + 0.000632524
    [70]	cv_agg's auc: 0.976419 + 0.000650458
    [80]	cv_agg's auc: 0.977098 + 0.000745719
    [90]	cv_agg's auc: 0.97755 + 0.000818675
    [100]	cv_agg's auc: 0.977911 + 0.00090971
    [110]	cv_agg's auc: 0.978179 + 0.000769916
    [120]	cv_agg's auc: 0.978444 + 0.000752198
    [130]	cv_agg's auc: 0.978642 + 0.00075305
    [140]	cv_agg's auc: 0.978788 + 0.000703718
    [150]	cv_agg's auc: 0.978902 + 0.000765816
    [160]	cv_agg's auc: 0.979049 + 0.000728942
    [170]	cv_agg's auc: 0.979122 + 0.000753131
    [180]	cv_agg's auc: 0.979211 + 0.00080433
    [190]	cv_agg's auc: 0.979264 + 0.000772557
    [200]	cv_agg's auc: 0.979262 + 0.000780631
    best_score: 0.9792841834551915 best_iteration: 198



```python
model = lgb.train(lgb_params,
                  lgb.Dataset(train_x, train_y),
                  num_boost_round=best_lgb_iteration
                  )

preds = model.predict(test_x)

sub_df = pd.DataFrame({'click_id': ids, 'is_attributed': preds})
sub_df.to_csv("lgb_starter.csv", index=False)
```


```python
sub_df.shape
```




    (18790469, 2)


