# 线性回归

```python
#生成数据
data_raw = pd.read_csv('./dnn_gbdt.csv',header=None,sep=' ',names=['label', 'dnn_p', 'gbdt_p'])
data_raw.count()
data_raw.head()
featrues = ['dnn_p','gbdt_p']
X = data_raw[featrues]
Y = data_raw['label']

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
#训练
reg = linreg.fit(X, Y)
#打印训练参数
print linreg.intercept_
print linreg.coef_
#预测
Y_pre = reg.predict(X)
#计算AUC
roc_auc_score(Y, Y_pre)
```



