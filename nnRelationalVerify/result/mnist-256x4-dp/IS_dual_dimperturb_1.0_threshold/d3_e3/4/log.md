## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.7344323225406553


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692)
1: (-0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081)
2: (-0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965)
3: (-0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550)
4: (-0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409)
5: (-0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894)
6: (-0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582)
7: (-0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213)
8: (-0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886)
9: (-0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.43 + 3.28 = 4.71 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -0.5775032, upper bound: 0.5775021
