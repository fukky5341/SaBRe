## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 8)
Time budget: 600 seconds
Split limit: 100
Threshold: 3.67655567


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-2.2437325, 1.7628641, -2.2437325, 1.7628641, -4.0065966, 4.0065966)
1: (-1.7677033, 1.6165466, -1.7677033, 1.6165466, -3.3842499, 3.3842499)
2: (-2.7652001, 1.6519299, -2.7652001, 1.6519299, -4.4171300, 4.4171300)
3: (-2.2114301, 1.5447328, -2.2114301, 1.5447328, -3.7561629, 3.7561629)
4: (-2.4180388, 1.7129834, -2.4180388, 1.7129834, -4.1310225, 4.1310225)
5: (-1.9751782, 1.7729955, -1.9751782, 1.7729955, -3.7481737, 3.7481737)
6: (-2.1666057, 1.8631227, -2.1666057, 1.8631227, -4.0297284, 4.0297284)
7: (-2.2731884, 1.8224580, -2.2731884, 1.8224580, -4.0956464, 4.0956464)
8: (-2.5782835, 2.3646996, -2.5782835, 2.3646996, -4.9429832, 4.9429832)
9: (-1.9551835, 2.1922305, -1.9551835, 2.1922305, -4.1474142, 4.1474142)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.91 + 2.93 = 4.83 seconds
status: Status.VERIFIED
relational distance
Output dim: 2, lower bound: -3.8700586, upper bound: 3.8700586
