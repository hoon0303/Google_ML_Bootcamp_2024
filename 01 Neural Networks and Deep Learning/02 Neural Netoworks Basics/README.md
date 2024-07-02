# Deep Learning 기초 정리

## 목차
1. 개요
2. 로지스틱 회귀
3. 비용 함수와 손실 함수
4. 경사 하강법
5. 계산 그래프
6. 벡터화

---

### 1. 개요
- **딥러닝(Deep Learning)**: 뉴럴 네트워크(Neural Networks)를 훈련하는 기술로, 복잡하고 큰 규모의 뉴럴 네트워크를 훈련하여 문제를 해결합니다.
- **뉴럴 네트워크(Neural Network)**: 입력 데이터와 출력을 연결하는 복잡한 함수. 인간의 뇌 구조를 모방한 것으로 다층 퍼셉트론(multi-layer perceptron)으로 구성됩니다.

### 2. 로지스틱 회귀
- **로지스틱 회귀(Logistic Regression)**: 이진 분류 문제에서 사용하는 학습 알고리즘. 입력 특징 벡터 X가 주어졌을 때, Y가 1일 확률을 예측하는 모델입니다.
- **시그모이드 함수(Sigmoid Function)**: 로지스틱 회귀의 출력 Y_hat을 0과 1 사이의 값으로 변환하는 함수. Z = W^T X + B를 입력으로 받아 1 / (1 + e^(-Z))를 출력합니다.
- **입력 데이터**: 예를 들어, 이미지 데이터가 64x64 픽셀이고 RGB 채널을 갖는다면, 각 이미지는 64x64x3 = 12288 차원의 벡터로 변환됩니다.

### 3. 비용 함수와 손실 함수
- **비용 함수(Cost Function)**: 모델의 예측 값(Y_hat)과 실제 값(Y) 사이의 차이를 측정하는 함수. 로지스틱 회귀에서 사용하는 비용 함수는 로그 손실(log loss) 함수입니다.
- **손실 함수(Loss Function)**: 단일 훈련 예제에서의 비용을 계산하는 함수. 로지스틱 회귀에서는 다음과 같이 정의됩니다:
  - L(Y_hat, Y) = - (Y * log(Y_hat) + (1 - Y) * log(1 - Y_hat))
- **전체 비용 함수**: 모든 훈련 예제에 대한 평균 손실을 계산합니다:
  - J(W, B) = (1 / m) * Σ L(Y_hat^(i), Y^(i))
- **목표**: 비용 함수 J(W, B)를 최소화하여 모델의 예측 성능을 향상시킵니다.

### 4. 경사 하강법
- **경사 하강법(Gradient Descent)**: 비용 함수를 최소화하기 위해 사용되는 최적화 알고리즘. 파라미터 W와 B를 업데이트하여 비용 함수 J(W, B)를 최소화합니다.
- **업데이트 규칙**:
  - W := W - α * (dJ / dW)
  - B := B - α * (dJ / dB)
  - 여기서 α는 학습률(learning rate)입니다.
- **단계**:
  - 1. **초기화**: 파라미터 W와 B를 초기화합니다.
  - 2. **순전파**: 현재 파라미터를 사용하여 예측 값을 계산합니다. (Y_hat = sigmoid(W^T X + B))
  - 3. **손실 계산**: 예측 값(Y_hat)과 실제 값(Y) 사이의 손실을 계산합니다.
  - 4. **역전파**: 손실에 대한 그래디언트를 계산합니다.
    - dZ = Y_hat - Y
    - dW = (1/m) * X * dZ^T
    - dB = (1/m) * Σ dZ
  - 5. **파라미터 업데이트**: 그래디언트를 사용하여 파라미터를 업데이트합니다.
  - 6. **반복**: 수렴할 때까지 2~5단계를 반복합니다.
- **예시 코드**:
  ```python
  for i in range(num_iterations):
      Z = np.dot(W.T, X) + B
      Y_hat = sigmoid(Z)
      loss = - (Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))
      cost = (1/m) * np.sum(loss)
      
      dZ = Y_hat - Y
      dW = (1/m) * np.dot(X, dZ.T)
      dB = (1/m) * np.sum(dZ)
      
      W = W - alpha * dW
      B = B - alpha * dB

### 5. 계산 그래프
- **계산 그래프(Computation Graph)**: 연산을 노드와 에지로 표현하여 순전파(forward pass)와 역전파(backpropagation) 과정을 시각화한 그래프.
- **순전파(Forward Pass)**: 입력 데이터를 통해 예측 값을 계산하는 과정.
- **역전파(Backward Pass)**: 예측 값과 실제 값 사이의 오차를 기반으로 가중치의 그래디언트를 계산하여 파라미터를 업데이트하는 과정.
- **예시**:
  - 주어진 함수 J(a, b, c) = 3(a + bc)의 계산 그래프:
    1. u = bc
    2. v = a + u
    3. J = 3v

```python
# 계산 그래프 예시
import numpy as np

# 초기 값 설정
a = 5
b = 3
c = 2

# 순전파
u = b * c
v = a + u
J = 3 * v

# 역전파
dJ_dv = 3
dv_da = 1
dv_du = 1
du_db = c
du_dc = b

dJ_da = dJ_dv * dv_da  # 3
dJ_du = dJ_dv * dv_du  # 3
dJ_db = dJ_du * du_db  # 6
dJ_dc = dJ_du * du_dc  # 9

print(f"dJ/da: {dJ_da}, dJ/db: {dJ_db}, dJ/dc: {dJ_dc}")
```
### 6. 벡터화
- **벡터화(Vectorization)**: 명시적 for 루프를 사용하지 않고 벡터 및 행렬 연산을 통해 알고리즘을 구현하는 기술.
- **장점**:
  - 코드의 간결성과 가독성 향상.
  - 대규모 데이터셋에 대한 효율적인 연산 가능.
- **예시**: 로지스틱 회귀의 경사 하강법을 벡터화하여 구현하는 방법.
- **벡터화된 경사 하강법 예시**:
```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 입력 데이터와 초기 파라미터 설정
np.random.seed(1)  # 결과 재현성을 위해 시드 설정
X = np.random.rand(12288, 1000)  # 1000개의 64x64x3 이미지 데이터
Y = np.random.randint(0, 2, (1, 1000))  # 0 또는 1의 이진 레이블
W = np.random.rand(12288, 1)
B = np.random.rand()
alpha = 0.01
num_iterations = 1000

for i in range(num_iterations):
    # 순전파
    Z = np.dot(W.T, X) + B
    Y_hat = sigmoid(Z)
    
    # 손실 계산
    loss = - (Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))
    cost = (1 / X.shape[1]) * np.sum(loss)
    
    # 역전파
    dZ = Y_hat - Y
    dW = (1 / X.shape[1]) * np.dot(X, dZ.T)
    dB = (1 / X.shape[1]) * np.sum(dZ)
    
    # 파라미터 업데이트
    W = W - alpha * dW
    B = B - alpha * dB

print(f"최종 비용: {cost}")
