import numpy as np
import pandas as pd
from sklearn import linear_model
import seaborn as sns


beta = [0.2]
x1 = np.random.random(size=1000) * 10
x = np.transpose([x1])
# y = 0.2 * x1 + b0
y = np.dot(x, beta) + np.random.normal(scale=0.5, size=1000)

df = pd.DataFrame(dict(x1=x1, y=y))

# sns.lmplot("x1", "y")

## sklearnのlinear model
clf = linear_model.LinearRegression()
clf.fit(x, y)
print("sklearn's linear_model calculated coef: {}".format(clf.coef_))

## numpyの最小二乗法
A = np.array([x1,np.ones(len(x1))])
A = A.T
a, b = np.linalg.lstsq(A, y)[0]
print("numpy's least square calculated coef: {}".format(a))
print("numpy's least square calculated intercept: {}".format(b))

## 泥臭く求める

### xに切片項を加える必要がある
x = np.array([x1,np.ones(len(x1))]).T
### p.15 exp.2.6
coef = np.dot(np.dot(np.linalg.inv(np.dot(x.T, x)) ,x.T), y.reshape((1000, 1)))

print("self calculated coef: {}".format(coef[0]))
print("self caculated intercept:{}".format(coef[1]))
