import tensorflow as tf

# 선형회귀 모델(Wx + b)을 위한 tf.Variable을 선언
W = tf.Variable(tf.random.normal(shape=[1]))
b = tf.Variable(tf.random.normal(shape=[1]))

@tf.function
def linear_model(x):
  return W*x + b

# 손실 함수(Cost Function)를 정의
# MSE 손실함수 \mean{(y' - y)^2}
@tf.function
def mse_loss(y_pred, y):
  return tf.reduce_mean(tf.square(y_pred - y))

# 최적화를 위한 그라디언트 디센트 옵티마이저를 정의(learning rate = 0.01)
optimizer = tf.optimizers.SGD(0.01)

# 최적화를 위한 function을 정의
@tf.function
def train_step(x, y):
  with tf.GradientTape() as tape:
    # 선형회귀 모델의 예측값 계산
    y_pred = linear_model(x)

    # 손실 함수 계산
    loss = mse_loss(y_pred, y)

    # 그라디언트 계산
  gradients = tape.gradient(loss, [W, b])
  optimizer.apply_gradients(zip(gradients, [W, b]))

# 트레이닝을 위한 입력값과 출력값을 준비
x_train = [1, 2, 3, 4]
y_train = [2, 4, 6, 8]

# 경사하강법을 1000번 수행
for i in range(1000):
  train_step(x_train, y_train)

# 테스트를 위한 입력값을 준비
x_test = [3.5, 5, 5.5, 6]
# 테스트 데이터를 이용해 학습된 선형회귀 모델이 데이터의 경향성(y=2x)을 잘 학습했는지 측정
# 예상되는 참값 : [7, 10, 11, 12]
print(linear_model(x_test).numpy())