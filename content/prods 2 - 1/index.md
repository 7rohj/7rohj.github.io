---
emoji: 🚫
title: 선형회귀(2)
date: '2022-04-05 03:00:00'
author: 강화정
tags: 공부 자격증 빅데이터 모델링
categories: 빅분기
---

선형회귀분석의 기본적인 가정이 있다.
선형회귀분석을 하기 전에 검토해봐야할 몇 가지 가정.

|||
|:---:|:---|
|선형성|독립변수와 종속변수가 선형적이어야 한다.|
|잔차 정규성|잔차의 기댓값은 0이며 정규분포를 이루어야 한다.|
|잔차 독립성| 잔차들은 서로 독립적이어야 한다|
|잔차 등분산성|잔차들의 분산이 일정해야 한다.|
|다중 공산성(VIF)| 다중 회귀분석을 수행할 경우 3개 이상의 독립변수 간에 상관관계로 인한 문제가 없어야 한다.|


**[아래는 파이썬을 통해 다중공산성 확인 및 변수 제거 실습한 이미지 이다]**<br/>
데이터는 `보스턴 주택 데이터`
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

![사진](./[ols_result.png])



<br/>
<br/>
<br/>






