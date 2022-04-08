---
emoji: 🎢
title: 전복 나이 예측 경진대회 
date: '2022-04-09 00:00:00'
author: 강화정
tags: 공부 빅데이터 데이콘 대회 예측
categories: 🤍데이콘🤍
---

<br/>
오! 데이콘에서 실시한 대회에서 2분동안 3등을 했었다. (600명 조금 넘는 참가자 중에서!)<br/>
영광의 캡쳐 .... ✨

### 2분간 3등.png ^.^

![picture](https://github.com/7rohj/7rohj.github.io/blob/2aed1e1cc54c249b0a06c928209354c821392d43/content/dacon%20contest/2%EB%B6%84%EA%B0%843%EB%93%B1.png?raw=true)
> 오잉 내가 내 이름까지 가려버렸네 

최종 결과는 public 10등/private 22등을 했다 :)<br/>
정말 신기하고 재미있는 경험이였다 ㅎㅎㅎㅎㅎㅎ<br/>
이번 `전복 나이 예측 경진대회`를 진행하면서 많이 배우고 가는 것 같다. <br/>

이번 경진대회를 통해 새로 알게 된건 `Auto ML` 이다.
이게 **pycaret 라이브러리**를 통해 머신러닝을 이용하는 것인데, 하이퍼파라미터만 내가 수정하면 되는거라
손이 별로 안가고 무척 편리했다! <br/> 내가 어떤 모델을 이용해서 진행할 것인지 처음 선택할 때 도움이 많이 되었다.<br/>
(마지막에는 pycaret을 이용하진 않았지만 ^^...)

## 아무튼 코드 설명 시작 ~~!

### 데이터 탐색

![picture](https://github.com/7rohj/7rohj.github.io/blob/2aed1e1cc54c249b0a06c928209354c821392d43/content/dacon%20contest/images1.PNG?raw=true)

`train data set`은 id, Gender, Lenght, Diameter, Height, Whole Weight, Shucked Weight, Viscra Weight, Shell Weight, Target의 칼럼을 갖고 있고
`test data set`은 Target을 제외한 나머지 칼럼들을 갖고 있다. shape은 (1253, 10) 그리고 (2924, 9) 로 test data가 더 많은 row를 갖고 있는 양상을 보여줬다. 

<br/>

![picture](https://github.com/7rohj/7rohj.github.io/blob/2aed1e1cc54c249b0a06c928209354c821392d43/content/dacon%20contest/images2.PNG?raw=true)

범주형 칼럼인 `Gender 칼럼`을 `value_counts` 를 통해 얼마나 어떻게 분포 되어 있는지 확인했다.
train이나 test에서 M이 다른 I과 F 보다 약 5퍼센트 포인트 더 많았다.

<br/>

![picture](https://github.com/7rohj/7rohj.github.io/blob/2aed1e1cc54c249b0a06c928209354c821392d43/content/dacon%20contest/images3.PNG?raw=true)

다행히 null 값은 없었다 ㅎㅎ

<br/>

![picture](https://github.com/7rohj/7rohj.github.io/blob/2aed1e1cc54c249b0a06c928209354c821392d43/content/dacon%20contest/images4.PNG?raw=true)
![picture](https://github.com/7rohj/7rohj.github.io/blob/2aed1e1cc54c249b0a06c928209354c821392d43/content/dacon%20contest/images5.PNG?raw=true)

distplot 으로 확인한 Target의 히스토그램 🙄
평균적으로 10대에 분포 되어 있었고 29살의 outlier도 boxplot을 이용해 확인할 수 있었다.
(아웃라이어는 삭제하는게 이롭다고 생각했었는데 삭제하는 것보다 삭제 하지 않은게 mae가 더 낮게 나왔다.
아마 test data의 Target 값은 20대에 고루 분포되어 있었던건 아닐까? 생각해봤다 ㅎㅎ
그래서 copy 해서 15세 이상인 친구들의 Target 값을 1이나 2를 더하는 전처리도 수행해봤는데 하지 않는게 결과가 더 좋게 나왔다. :))

<br/>

### `train/test boxplot`

![picture](https://github.com/7rohj/7rohj.github.io/blob/2aed1e1cc54c249b0a06c928209354c821392d43/content/dacon%20contest/images6.PNG?raw=true)
![picture](https://github.com/7rohj/7rohj.github.io/blob/2aed1e1cc54c249b0a06c928209354c821392d43/content/dacon%20contest/images6-2.PNG?raw=true)

![picture](https://github.com/7rohj/7rohj.github.io/blob/2aed1e1cc54c249b0a06c928209354c821392d43/content/dacon%20contest/images7.PNG?raw=true)

Lenght, Diameter, Height 비슷 <br/>
나머지것들은 그것대로 비슷한 분포

데이터 분석은 간단하게 여기서 마치고 ~~~~ 위에서 설명했던
**pycaret 라이브러리** 에 대해서 ! 공부해보도록 하자 ^.^

<br/>
<br/>
<br/>

