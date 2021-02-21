### LOG: 20210124 - (RESEARCH) WNV prediction NN data modeling (NN v2)
https://www.kaggle.com/jhinchoh/wnv-prediction-nn-v2


```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../../../data_other/kaggle_data/predict_west_nile_virus/result_log_20210124.csv',header=None,delimiter=':')
df.reset_index(inplace=True)
df.columns = ['group','name','value']
df['group'] = df['group'] // 3

df = df.pivot(index='group', columns='name', values='value') \
       .reset_index().rename_axis(None, axis=1).drop('group',axis=1).set_index('Params')

display(df)
df.columns = ['Train AUC','Valid AUC']
df = df.astype('float')
df.plot()
plt.xticks(rotation=90)
plt.show()
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
      <th>Train AUC</th>
      <th>Valid AUC</th>
    </tr>
    <tr>
      <th>Params</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>hidden=600,dropout=0.3,CULEX PIPIENS/RESTUANS</th>
      <td>0.8453521728515625</td>
      <td>0.8256840705871582</td>
    </tr>
    <tr>
      <th>hidden=600,dropout=0.3,CULEX RESTUANS</th>
      <td>0.8855190873146057</td>
      <td>0.8645884990692139</td>
    </tr>
    <tr>
      <th>hidden=600,dropout=0.3,CULEX PIPIENS</th>
      <td>0.8203920125961304</td>
      <td>0.7928934097290039</td>
    </tr>
    <tr>
      <th>hidden=600,dropout=0.2,CULEX PIPIENS/RESTUANS</th>
      <td>0.8966070413589478</td>
      <td>0.8443787693977356</td>
    </tr>
    <tr>
      <th>hidden=600,dropout=0.2,CULEX RESTUANS</th>
      <td>0.9213939905166626</td>
      <td>0.8615221977233887</td>
    </tr>
    <tr>
      <th>hidden=600,dropout=0.2,CULEX PIPIENS</th>
      <td>0.8809412717819214</td>
      <td>0.773032009601593</td>
    </tr>
    <tr>
      <th>hidden=600,dropout=0.2,CULEX TERRITANS</th>
      <td>0.9244016408920288</td>
      <td>0.9510204195976257</td>
    </tr>
    <tr>
      <th>hidden=600,dropout=0.2,CULEX SALINARIUS</th>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>hidden=600,dropout=0.2,the rest of the group including CULEX SALINARIUS</th>
      <td>0.9937500953674316</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>hidden=600,dropout=0.2,the rest of the group including CULEX SALINARIUS and CULEX TERRITANS</th>
      <td>0.9690346121788025</td>
      <td>0.5846645832061768</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](/img/log_20210124_output_1.png)
    



```python

```
