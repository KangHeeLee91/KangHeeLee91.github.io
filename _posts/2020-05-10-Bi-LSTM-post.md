---
title: "Bi-LSTM 이론 파헤치기!!!"
date: 2020-05-10 03:18:00 -0400
categories: Deep-Leaning
---

## 3. Bi-LSTM?
### 3-1. Bi-LSTM 이란?

![image](https://user-images.githubusercontent.com/31266360/81478414-40ca5280-9258-11ea-9e1a-6cd20aaa5a9a.png)

위 그림은 Bi-LSTM의 구조이다.
Bi-LSTM은 Bidirectional LSTM으로 양방향성을 갖는 LSTM이라고 볼 수 있다.
  위 그림을 보면 주황색 블록의 forward 정방향 LSTM이 있고 초록색 블록의 backward 역방향 LSTM이 있다.
  
  Tokenizing과 Embedding 과정이 완료된 Sequence 입력이 들어오면 forward 방향의 LSTM과 backward 방향의 LSTM에 각각 입력된다.
  
  이후 각각 forward 방향 LSTM의 t번째 Cell과 backward 방향의 LSTM의 N-t번째 Cell의 은닉값을 Concat을 한다.
  이후 모든 Concat 결과물을 가지고 최종 출력을 뽑게 된다.
  위 그림과 같이 Classification에 Bi-LSTM을 사용하면 최종 출력으로 각 LSTM Cell들의 Concat 결과물들을 다시 Concat한 후 Softmax함수를 통해 Class를 추론하게 된다.

### 3-2. Bi-LSTM 실습
