---
emoji: 🎢
title: 전복 나이 예측 경진대회 
date: '2022-04-09 00:00:00'
author: 강화정
tags: 공부 빅데이터 데이콘 대회 예측
categories: 🤍데이콘🤍
---

오! 데이콘에서 실시한 대회에서 2분동안 3등을 했었다. (600명 조금 넘는 참가자 중에서!)<br/>
영광의 캡쳐 .... ✨<br/>
(사진 넣기)

최종 결과는 public 10등/private 22등을 했다 :)<br/>
정말 신기하고 재미있는 경험이였다 ㅎㅎㅎㅎㅎㅎ<br/>
이번 `전복 나이 예측 경진대회`를 진행하면서 많이 배우고 가는 것 같다. <br/>

이번 경진대회를 통해 새로 알게 된건 `Auto ML` 이다.
이게 **pycaret 라이브러리**를 통해 머신러닝을 이용하는 것인데, 하이퍼파라미터만 내가 수정하면 되는거라
손이 별로 안가고 무척 편리했다! <br/> 내가 어떤 모델을 이용해서 진행할 것인지 처음 선택할 때 도움이 많이 되었다.<br/>
(마지막에는 pycaret을 이용하진 않았지만 ^^...)

## 아무튼 코드 설명 시작 ~~!

### 데이터 탐색
train data set은 id, Gender, Lenght, Diameter, Height, Whole Weight, Shucked Weight, Viscra Weight, Shell Weight, Target의 칼럼을 갖고 있고
test data set은 Target을 제외한 나머지 칼럼들을 갖고 있다. shape은 (1253, 10) 그리고 (2924, 9) 로 test data가 더 많은 row를 갖고 있는 양상을 보여줬다.




<br/>
