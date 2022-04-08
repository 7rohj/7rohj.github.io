---
emoji: 🌱
title: 데이터 사전처리(Preprocessing)
date: '2022-04-08 00:00:00'
author: 강화정
tags: 공부 자격증 빅데이터 전처리
categories: 빅분기
---

일단 null 값이 있는 row는 drop 해주고,
회귀 모델을 적용하기 전에 타깃 값의 분포도가 정규 분포인지 확인한다. (drop 말고도 다양한 방법이 있음. 다음에 다뤄보도록 해보자)

```bash
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

house_df_org = pd.read_csv('house_price.csv')
house_df = house_df_org.copy() # 같은 데이터 프레임 복사.
house_df.head(3)
```



```bash
print('데이터 세트의 Shape:', house_df.shape)
print('\n전체 피처의 type \n', house_df.dtypes.value_counts())
isnull_series=house_df.isnull().sum()
print('\nNull 칼럼과 그 건수:\n', isnull_series[isnull_series>0].sort_values(ascending=False))
```

결과는 생략 `o.~`

<br/>
<br/>

회귀 모델을 적용하기 잔에 타깃 값의 분포도가 정규 분포인지 확인해야 한다.
위 데이터는 데이터 값의 분포가 중심에서 왼쪽으로 치우친 형태로 정규 분포에서 벗어나 있다.

```bash
plt.title('Original Sale Price Histogram')
sns.distplot(trainDF['SalePrice'])
```

<br/>
<br/>

### POINT

`정규 분포가 아닌 결과값`을 정규 분포 형태로 변환하기 위해 `로그 변환(Log Transformation)`을 적용한다.
**넘파이의 `log1p()`를 이용해 로그 변환한 결괏값을 기반으로 학습한 뒤, 예측 시에는 다시 결괏값을 `expm1()`으로 추후에 환원**하면 된다.
```bash
plt.title('Log Transformed Sale Price Histogram')
log_SalePrice = np.log1p(house_df['salePrice'])
sns.distplot(log_SalePrice)
```

<br/>
<br/>

그리고 문자형 피처는 `get_dummies()`를 이용해준다. 만약 Gender 라는 칼럼이 있으면 Gender_M, Gender_F 하고 해당되는 부분에 1이라는 값이 주어지는 형태의 테이블로 바뀜
<br/>
<br/>
<br/>
