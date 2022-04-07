---
emoji: 🚫
title: 선형회귀(2)
date: '2022-04-05 03:00:00'
author: 강화정
tags: 공부 자격증 빅데이터 모델링
categories: 빅분기
---

선형회귀분석의 기본적인 가정이 있다.
`선형회귀분석을 하기 전에 검토해봐야할 몇 가지 가정.`

|||
|:---:|:---|
|선형성|독립변수와 종속변수가 선형적이어야 한다.|
|잔차 정규성|잔차의 기댓값은 0이며 정규분포를 이루어야 한다.|
|잔차 독립성| 잔차들은 서로 독립적이어야 한다|
|잔차 등분산성|잔차들의 분산이 일정해야 한다.|
|다중 공선성(VIF)| 다중 회귀분석을 수행할 경우 3개 이상의 독립변수 간에 상관관계로 인한 문제가 없어야 한다.|

<br/>
<br/>
<br/>

**[아래는 파이썬을 통해 다중공산성 확인 및 변수 제거 실습한 이미지 이다]**<br/>
데이터는 `보스턴 주택 데이터`<br/>

## 〰 1 〰
``` bash
import pandas as pd 
import numpy as np
import statsmodels.api as sm

# 데이터 불러오기
boston = pd.read_csv("./Boston_house.csv")
boston_data = boston.drop(['Target'], axis=1)

# crim, rm, lstat을 통한 다중 선형회귀분석
x_data = boston[["CRIM","RM","LSTAT"]] #변수 여러개
target = boston[["Target"]]

# for b0, 상수항 추가
x_data1 = sm.add_constant(x_data, has_constant = "add")

# OLS 검정
multi_model = sm.OLS(target, x_data1)
fitted_multi_model = multi_model.fit()
fitted_multi_model.summary()
```

### `"CRIM","RM","LSTAT"` OLS 결과

![picture](https://github.com/7rohj/7rohj.github.io/blob/4b2a9b2c79038944080b02ea7a44705979f6f415/content/prods%202%20-%201/olsresult.png?raw=true)

<br/>
<br/>
<br/>

## 〰 2 〰
``` bash
## boston data에서 원하는 변수만 뽑아오기
x_data2 = boston[['CRIM','RM', 'LSTAT', 'B', 'TAX', 'AGE', 'ZN', 'NOX', 'INDUS']]
x_data2.head()

# 상수항 추가
x_data2_ = sm.add_constant(x_data2, has_constant = "add")

# 회귀모델 적합
multi_model2 = sm.OLS(target, x_data2_)
fitted_multi_model2 = multi_model2.fit()

# 결과 출력
fitted_multi_model2.summary()
```

### `FULL FEATURES` OLS 결과 

![picture](https://github.com/7rohj/7rohj.github.io/blob/7f3956c123d04a5f9b1539c99feefc038c66455e/content/prods%202%20-%201/olsresult2.png?raw=true)

*Warnings 에서의 2번 항목이 생겼다. 다중공산성 주의!
> 강한 다중공선성 또는 다른 numerical 문제가 발생했다고 암시.

<br/>
<br/>
<br/>

## 〰 3 〰
``` bash
# 변수끼리 산점도를 시각화
sns.pairplot(x_data2)
plt.show()
```

### `sns.pairplot` 결과
![picture](https://github.com/7rohj/7rohj.github.io/blob/7f3956c123d04a5f9b1539c99feefc038c66455e/content/prods%202%20-%201/pairplot.png?raw=true)

그림에는 없지만 heatmap을 이용해 상관 matrix를 확인했을때 `0.5가 넘어가는 변수들간의 상관관계`가 빈출되는 것은
충분히 `다중공선성 발생`을 의심할 수 있다. 즉, 그 변수들은 제거해 주는게 이롭다고 판단할 수 있는 것이다.
위 그림에서 보이듯 음 또는 양의 상관관계를 나타내는 그래프들의 변수들 또한 그렇다고 얘기할 수 있다.

<br/>
<br/>
<br/>

## 〰 4 〰
```bash
from statsmodels.stats.ouliers_influence import variance_inflation_factor 

# VIF사용을 위한 라이브러리, statsmodels안에 존재한다.
# 사실 모든 통계기법이 statsmodels 모듈에 존재하여 
# 이 중에 필요한 통계기법을 찾아 import를 진행하면 된다.

vif = pd.DataFrame()
vif["VIF Factor"] = [varinace_inflation_factor(x_data2.values, i) for i in range(x_data2.shape[1])]
vif["features"] = x_data4.columns
vif
```





