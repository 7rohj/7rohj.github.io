---
emoji: 5️⃣
title: 정보처리기사 실기 Day5
date: '2022-04-05 00:00:00'
author: 강화정
tags: 정처기 공부 실기 자격증 회고
categories: 자격증
---
<br/>
실기 접수를 끝마쳤다 ㅎㅎ 사전입력을 했다는 것을 까먹고 처음부터 하고 있었는데
중간에 사전입력했다고 마이페이지에 가봐라는 창이 뜨며 초기화면으로 넘어가길래 당황했었다 ^.^...<br/>
한 페이지 넘어가는데에 5분~6분 정도 걸렸었는데... 미리 좀 말해주지 😩<br/>
아무튼 20분 정도 걸려서 시험 접수를 잘 마치긴했다 ! 접수된 시험들이 많군.........<br/>
<br/>
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ 🚀<br/>
<br/>
<br/>
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀🚀<br/>
<br/>
<br/>
끝까지 힘내즈앗 얍얍얍 🚀<br/>
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀🌏<br/>
<br/>
<br/>

#  S T A R T
## 2020년 제2회 기출복원문제
### 16. 리눅스 커널을 기반으로 동작하며 `자바`와 `코틀린 언어`로 개발된 핸드폰이나 소형기기에 사용되는 오픈소스 플랫폼인 `모바일 운영체계`는 무엇인지 쓰시오.
`안드로이드`
<br/>
<br/>
<br/>
<br/>

### 17. 리눅스 서버에 a.txt라는 파일이 있다. 다음 <조건>에 알맞은 명령문을 쓰시오.

<조건>
> - 사용자에게는 읽기, 쓰기, 실행의 세 개의 권한을 모두 부여한다.
> - 그룹에게는 읽기, 실행 두 개의 권한을 부여한다.
> - 그룹 외 사용자에게는 실행 권한을 부여한다.
> - 한 줄로 명령문이 작성되어야 하며, 아라비안 숫자를 사용하여 8진수로 권한을 부여한다.

`chmod 751 a.txt`

**[추가정보]**<br/>
**chmod 777 b.txt** // b.txt 파일 `모든 사용자`에게 read, write, execute 권한 부여<br/>
**chmod 751 b.txt** // b.txt 파일 `user` 는 모든 권한, `group` 은 read, execute 권한, `other` 는 execute 권한 부여<br/>
**chmod 700 b.txt** // b.txt 파일 `user` 는 모든 권한 `나머지`는 권한 제거
<br/>
<br/>
<br/>
<br/>

### 18. 다음은 IP 인프라 서비스 관리 실무와 관련된 <실무 사례>에 대한 설명이다. 빈칸 (   ) 안에 가장 적합한 용어를 한글 또는 영문으로 쓰시오.
<실무 사례>
> ...(생략)... 데이터 백업 솔루션은 만일의 사태에 대비하여 시스템 내의 데이터 유실을 방지하고, 서비스의 연속성을 보장하는 목적을 가지고 어떤 상황에서도 계획된 (   )과 목표 복구 시점을 보장해야
> 할 수 있는 제품이어야 한다. (   )는 시스템 장애와 같은 상황에서의 `비상사태 또는 업무중단 시점부터 ***업무가 복구되어 다시 정상가동 될 때까지의 시간**`을 의미하는 용어이다.
> RPO는 조직에서 발생한 여러 가지 재난 상황으로 IT 시스템이 마비되었을 때 각 업무에 필요한 데이터를 여러 백업 수단을 활용하여 복구할 수 있는 기준점을 의미한다. ...(생략)...
`RTO` 또는 `Recovery Time Objective`, 목표 복구 시간, 복구 목표 시간 ...
<br/>
<br/>
<br/>
<br/>

### 19. 다음 디자인 패턴과 관련된 설명에 가장 부합하는 용어를 영문으로 쓰시오.
- 디자인 패턴 중 (   )는 행위 패턴에 해당하며 1대다의 객체 의존관계를 정의한 것으로, 한 객체가 상태를 변화시켰을 때 의존관계에 있는 다른 객체들에게 자동으로 통지 알림이 전달되고 변경시킨다.
- (   )의 객체 간의 데이터 전달 방식은 푸시 방식과 풀 방식이 있으며, 기본적인 디자인의 원칙은 상호작용하는 객체 사이에서는 가능하면 결합도를 느슨하게 디자인하여 사용해야 한다.
`Observer` 또는 `Observer Pattertn`
<br/>
<br/>
<br/>
<br/>

### 20. 다음 신기술 동향과 관련된 설명에 가장 부합하는 용어를 영문 완전이름으로 쓰시오.
- (   )는 개방형 정부, 개방형 공공 데이터의 시대적 요구와 맞물려 있으며, 기존의 거대한 정보 생태계인 웹을 활용하고 웹 기술과 핵심 개념을 그대로 활용한다는 점에서 주목받고 있다.
- (   )의 주요 특징은 URI를 사용한다는 점이다. 흔히알고 있는 URL과 비슷한 개념으로 URL이 특정 정보 자원의 종류와 위치를 가리킨다면, URI는 HTTP 프로토콜을 통해 웹에 저장된 객체를 가리킨다는 점에서 다르다.
- 웹상에 존재하는 전세계 오픈된 정보를 하나로 묶는 RESTfull한 방식이며, 링크 기능이 강조된 시맨틱웹의 모형에 속한다고 볼 수 있다. 즉, (   )는 시맨틱웹을 실현시키기 위한 방법이자 기술적 접근점으로 볼 수 있다. <br/>
`Linked Open Data`
<br/>
<br/>
<br/>
<br/>

## 2020년 제3회 기출복원문제
### 1. `형상 통제`에 대해 간략히 설명하시오.
형상통제는 `형상에 대한 변경 요청이 있을 경우 변경 여부와 변경 활동을 통제하는 것`을 말한다. 변경된 요구사항에 대한 타당성을 검토하여 변경을 실행하고 그에 따라 변경된 산출물에 대한
버전관리를 수행하는 것이 형상 통제의 주요 활동이다. 즉, 형상통제는 소프트웨어 형상 변경 요청을 검토 승인하여 `현재의 베이스라인에 반영될 수 있도록 통제하는 것`을 의미한다.
<br/>
<br/>
<br/>
<br/>

### 2. EAI 구축 유형 중 Message Bus와 Hybrid를 제외한 빈칸 1~1에 해당하는 나머지 두가지 유형을 쓰시오.
(그림 삽입)
유형|설명
:---:|:---:
`Point to Point`|- 중간에 미들웨어를 두지 않고 각 애플리케이션 간 직접 연결<br/> - 솔루션 구매 없이 통합, 상대적 저렴하게 통합 가능<br/> - 변경, 재사용 어려움
`Hub&Spoke`|- 단일 접점이 허브 시스템을 통해 데이터를 전송하는 중앙 집중적 방식<br/> - 모든 데이터 전송 보장, 확장 및 유지 보수 용이 <br/> - 허브 장애 시 전체 영향

<br/>
<br/>
<br/>

### 3. UI는 사용자와 컴퓨터 상호 간의 소통을 원활히 할 수 있도록 도와주는 연계 작업을 뜻한다. U의 설계 원칙 중 직관성에 대해 간략히 설명하시오.
`누구나 쉽게 이해하고 사용할 수 있어야 한다.`
<br/>
<br/>
<br/>
<br/>

### 4. 다음 제어 흐름 그래프에 대한 분기 커버리지를 수행하는 경우의 테스트 케이스 경로를 7단계와 6단계로 나눠서 순서대로 나열하시오.
(그림삽입)<br/>
<br/>
<br/>
<br/>

⠀⠀⠀⠀⠀⠀⠀💘<br/>
⠀⠀⠀⠀⠀⠀‍🧍‍♀️<br/>
⠀⠀⠀⠀⠀⠀‍⠀‍⠀<span style="font-size:20%">🐈</span><br/>

⠀⠀⠀⠀⠀⠀⠀<span style="font-size:300%">🕳</span><br/>

⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀,⠀.............🐌
```toc

```
