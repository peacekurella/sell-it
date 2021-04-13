program: train.py
method: bayes
early_terminate:
  type: hyperband
  min_iter: 10
metric:
  goal: minimize
  name: Test/MSE
parameters:
  model:
    value: bodyAE
  learning_rate:
    distribution: log_uniform
    min: -9.2103403719
    max: -2.30258509292
  epochs:
    value: 200
  lmd:
    distribution: log_uniform
    min: -16
    max: 0.0
  batch_size:
    value: 512