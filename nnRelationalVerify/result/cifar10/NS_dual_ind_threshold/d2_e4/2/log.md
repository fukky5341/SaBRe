## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 2)
Time budget: 3600 seconds
Split limit: 100
Threshold: 0.8673587642119214


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1.6342446, 1.8813713, -1.6342446, 1.8813713, -3.0848260, 3.0848260)
1: (-5.0874925, 0.4877839, -5.0874925, 0.4877839, -5.0967975, 5.0967970)
2: (-0.9069280, 0.2799806, -0.9069280, 0.2799806, -0.6184200, 0.6184200)
3: (-0.9025570, 0.5434447, -0.9025570, 0.5434447, -1.4039502, 1.4039502)
4: (-3.3997905, -1.8272804, -3.3997905, -1.8272804, -0.9863646, 0.9863645)
5: (-1.9493551, -0.2465163, -1.9493551, -0.2465163, -1.5399034, 1.5399034)
6: (0.3500302, 2.2756245, 0.3500302, 2.2756245, -1.1569237, 1.1569238)
7: (-2.4252267, 0.9831107, -2.4252267, 0.9831107, -2.6365571, 2.6365571)
8: (-6.8932605, -2.8964016, -6.8932605, -2.8964016, -2.9264770, 2.9264770)
9: (-4.3776960, -0.9934518, -4.3776960, -0.9934518, -2.9217205, 2.9217205)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 9.50 + 214.21 = 223.71 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -0.7754149, upper bound: 0.7754037
