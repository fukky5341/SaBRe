## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.6442851107111194


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (9.3117714, 11.0277529, 9.3117714, 11.0277529, -1.0929978, 1.0929976)
1: (-18.8558655, -16.0991993, -18.8558655, -16.0991993, -2.0269022, 2.0269027)
2: (-3.4370537, -1.4411042, -3.4370537, -1.4411042, -1.4236751, 1.4236755)
3: (-13.0907698, -10.5306435, -13.0907698, -10.5306435, -1.7745891, 1.7745891)
4: (-15.1888399, -12.5623980, -15.1888399, -12.5623980, -1.7633996, 1.7633996)
5: (-5.8603497, -4.1480732, -5.8603497, -4.1480732, -0.9515619, 0.9515619)
6: (-3.3205850, -1.7698631, -3.3205850, -1.7698631, -1.1309075, 1.1309078)
7: (-7.1297069, -4.3905158, -7.1297069, -4.3905158, -2.4491024, 2.4491024)
8: (-2.5201378, -0.9655390, -2.5201378, -0.9655390, -1.1922536, 1.1922541)
9: (-8.9938412, -6.5298333, -8.9938412, -6.5298333, -1.6054664, 1.6054664)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.64 + 34.31 = 56.95 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -0.5165233, upper bound: 0.5165254
