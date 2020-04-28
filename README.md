## 블로그 소개

안녕하세요 저는 딥러닝 개발을 잘하고 싶은 새내기 개발자입니다. 이 블로그는 기초부터 응용까지 딥러닝 기반 NLP에 관한 내용들을 올릴 예정입니다. 모두들 함께 좋은 내용 만들어 갔으면 좋겠습니다.

### 커리큘럼


### LSTM
LSTM은 어디서 어떻게 생겨난 아이디어일까요?
아래의 바닐라 RNN을 한번 보도록합시다.
![valilaRNN](https://user-images.githubusercontent.com/31266360/80350874-52047e00-88ac-11ea-8af6-0ca9a15ce796.png)

기본적인 RNN의 모양은 위와 같습니다. 아주 단순하게 전 Cell에서 넘어온 Hidden state 값과 이번 Cell의 input 값을 concat한 후 Non-linear activation 함수를 통해 다음 Cell에 전파시킬 값을 반환하는 형태입니다.

![valilaRNNcal](https://user-images.githubusercontent.com/31266360/80352174-71040f80-88ae-11ea-8d6f-72e1d0f5eb91.png)

위의 공식과 같이 계산이 됩니다.

위와 같은 구조가 어떤 문제점이 있을까요?

![valilaRNNchain](https://user-images.githubusercontent.com/31266360/80352375-c17b6d00-88ae-11ea-9e3f-4d3ea0a6b032.png)

위 그림과 같이 RNN 셀이 여러개로 엮이게 된다면

![valilaRNNchaincal](https://user-images.githubusercontent.com/31266360/80352593-11f2ca80-88af-11ea-9aa3-3066aef54988.png)

맨 마지막 cell의 output값은 위와 같은 공식이 성립이 됩니다.
이 부분이 학습과정을 거치게 되면 역전파 과정을 거치며 tanh가 미분이 되게 되는데

![tanh](https://user-images.githubusercontent.com/31266360/80352894-8594d780-88af-11ea-9aba-e1482a300728.png)

위 그림과 같이 tanh의 도함수는 0에서 1사이의 값이 됩니다.
0에서 1사이의 값을 계속 곱연산을 하게되면 점점 값이 0에 수렴하는 방향으로 가게됩니다. 그렇게 되면 위처럼 RNN 셀이 길게 늘어진 경우 초반의 입력이 후반으로 갈 수록 0이라는 값에 가까워지게 때문에 정보를 잃게됩니다. 이렇게되면 초반 입력이 후반 Cell에 영향을 거의 미치지 못하게 된다는 것을 알 수 있습니다. 이러한 현상을 Vanishing gradient problem이라고 부릅니다. 기울기 값이 사라지는 문제입니다.

이러한 문제를 해결하기 위해 과거 Cell의 정보를 일정 필요한 만큼 기억해 나가자라는 개념에서
LSTM이 생겨났습니다.

그렇다면 LSTM은 어떻게 생겼을까요?

![LSTM](https://user-images.githubusercontent.com/31266360/80438660-aad12680-893f-11ea-8837-8ec26a110270.png)

위 그림과 같은 모양을 취하고 있습니다. 뭔가 복잡해 보이지만 제일 큰 변화는 Ct라는 과거 Cell state 정보를 따로 저장하는 변수가 생겼다는 점입니다.

그렇다면 LSTM의 각 Gate들이 어떠한 역할을 해서 어떻게 output 값을 도출해내는지 알아보도록 합시다.

처음으로 Forget gate를 볼 수 있습니다.

![forgetgate](https://user-images.githubusercontent.com/31266360/80440220-42844400-8943-11ea-8953-62e1efc42046.png)

위 그림을 보면 ft를 구하기 위해 ht-1과 xt를 concat한 후 특정 가중치를 곱연산하여 sigmoid 함수를 통해 값을 도출했습니다. 
이 ft는 현재 Cell의 입력들을 바탕으로 과거 Cell로부터 이어져온 정보들 중 어떤 정보를 얼마만큼 많이 남기고 어떤 정보를 얼마만큼 조금 남길것 인지를 정해주는 값입니다. Sigmoid 함수를 통과한 값은 0~1 사이의 값을 가지게 됩니다. 따라서 기존 정보에 sigmoid 함수를 통과한 값을 곱하면 그만큼 정보의 양이 줄어들게 되는 것 입니다. 예를 들어서 sigmoid를 통과한 값이 0.9다 라고 하면 해당 이전 정보를 0.9배 만큼 기억하게 되어 많이 기억하게 되고 sigmoid를 통과한 값이 0.1이라고 하면 해당 이전 정보의 0.1배 만큼만 기억하게 되어 조금 기억하게 되는 것입니다.
이전 정보의 형태가 [0.1, 2.0, -0.4, 0.5] 이라고 하고 sigmoid를 통과한 값이 [0.1, 0.9, 0.4, 0.9] 라고 한다면  이전 정보를 [0.01, 1.98, -0.16, 0.45] 만큼만 남기고 나머지는 까먹자! 라고 해당  Cell에서 정하게 되는 것입니다. 대신 정보를 까먹는 만큼 뒤에서 이번 셀에서 새로운 기억을 추가하는 과정이 있습니다. Forget gate에서는 sigmoid가 0~1 사이의 값을 출력으로 하기 때문에 어떤 데이터를 얼만큼 더 혹은 덜 기억하게 하는 역할을 한다는 것을 알 수 있었습니다. 


### Seq2Seq
