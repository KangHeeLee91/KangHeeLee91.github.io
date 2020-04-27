## 블로그 소개

안녕하세요 저는 딥러닝 개발을 잘하고 싶은 새내기 개발자입니다. 이 블로그는 기초부터 응용까지 딥러닝 기반 NLP에 관한 내용들을 올릴 예정입니다. 모두들 함께 좋은 내용 만들어 갔으면 좋겠습니다.

### 커리큘럼


### LSTM
LSTM은 어디서 어떻게 생겨난 아이디어일까?
아래의 바닐라 RNN을 한번 보도록하자
![valilaRNN](https://user-images.githubusercontent.com/31266360/80350874-52047e00-88ac-11ea-8af6-0ca9a15ce796.png)

기본적인 RNN의 모양은 위와 같다. 아주 단순하게 전 Cell에서 넘어온 Hidden state 값과 이번 Cell의 input 값을 concat한 후 Non-linear activation 함수를 통해 다음 Cell에 전파시킬 값을 반환하는 형태이다.

![valilaRNNcal](https://user-images.githubusercontent.com/31266360/80352174-71040f80-88ae-11ea-8d6f-72e1d0f5eb91.png)

위의 공식과 같이 계산이 된다.

위와 같은 구조가 어떤 문제점이 있을까?

![valilaRNNchain](https://user-images.githubusercontent.com/31266360/80352375-c17b6d00-88ae-11ea-9e3f-4d3ea0a6b032.png)

위 그림과 같이 RNN 셀이 여러개로 엮이게 된다면

![valilaRNNchaincal](https://user-images.githubusercontent.com/31266360/80352593-11f2ca80-88af-11ea-9aa3-3066aef54988.png)

맨 마지막 cell의 output값은 위와 같은 공식이 성립이 된다.
이 부분이 학습과정을 거치게 되면 역전파 과정을 거치며 tanh가 미분이 되게 되는데

![tanh](https://user-images.githubusercontent.com/31266360/80352894-8594d780-88af-11ea-9aba-e1482a300728.png)

위 그림과 같이 tanh의 도함수는 0에서 1사이의 값이 된다.
0에서 1사이의 값을 계속 곱연산을 하게되면 점점 값이 0에 수렴하는 방향으로 가게된다. 그렇게 되면 위처럼 RNN 셀이 길게 늘어진 경우 초반의 입력이 후반으로 갈 수록 0이라는 값에 가까워지게 때문에 정보를 잃게된다. 이렇게되면 초반 입력이 후반 Cell에 영향을 거의 미치지 못하게 된다는 것이다. 이러한 현상을 Vanishing gradient problem이라고 부른다. 기울기 값이 사라지는 문제이다.

이러한 문제를 해결하기 위해 과거 Cell의 정보를 일정 필요한 만큼 기억해 나가자라는 개념에서
LSTM이 생겨났다.


### 내용무
- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/KangHeeLee91/KangHeeLee91.github.io/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
