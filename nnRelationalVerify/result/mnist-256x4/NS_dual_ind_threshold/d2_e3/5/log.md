## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 5)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.097198775


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.8851730, 0.9936641, 0.8851730, 0.9936641, -0.1084911, 0.1084911)
1: (-0.0273599, 0.0013064, -0.0273599, 0.0013064, -0.0286663, 0.0286663)
2: (0.0010634, 0.0352309, 0.0010634, 0.0352309, -0.0341675, 0.0341675)
3: (-0.0112761, 0.0366631, -0.0112761, 0.0366631, -0.0479392, 0.0479392)
4: (-0.0253607, 0.0074003, -0.0253607, 0.0074003, -0.0327610, 0.0327610)
5: (0.0003263, 0.1017116, 0.0003263, 0.1017116, -0.1013853, 0.1013853)
6: (-0.0198776, 0.0185403, -0.0198776, 0.0185403, -0.0384178, 0.0384178)
7: (-0.0338814, 0.0112810, -0.0338814, 0.0112810, -0.0451624, 0.0451624)
8: (-0.0209270, 0.0345762, -0.0209270, 0.0345762, -0.0555032, 0.0555032)
9: (-0.0146789, 0.0279743, -0.0146789, 0.0279743, -0.0426532, 0.0426532)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.26 + 3.32 = 5.58 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -0.1023145, upper bound: 0.1023145
