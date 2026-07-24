## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0022162150620477694


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741)
1: (0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977)
2: (0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0107039, 0.0107039)
3: (-0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259)
4: (0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0016757, 0.0016757)
5: (-0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670)
6: (-0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753)
7: (-0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175)
8: (-0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897)
9: (1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.15 + 2.66 = 3.81 seconds
status: Status.VERIFIED
relational distance
Output dim: 9, lower bound: -0.0017606, upper bound: 0.0017606
