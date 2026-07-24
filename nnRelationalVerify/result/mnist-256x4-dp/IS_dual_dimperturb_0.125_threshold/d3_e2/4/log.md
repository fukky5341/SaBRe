## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0006444


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0040703, -0.0039495, -0.0040703, -0.0039495, -0.0000540, 0.0000540)
1: (0.0016163, 0.0022852, 0.0016163, 0.0022852, -0.0002989, 0.0002989)
2: (0.0098608, 0.0113553, 0.0098608, 0.0113553, -0.0006678, 0.0006678)
3: (0.0025492, 0.0031790, 0.0025492, 0.0031790, -0.0002814, 0.0002814)
4: (1.0066403, 1.0090835, 1.0066403, 1.0090835, -0.0010918, 0.0010918)
5: (0.0035332, 0.0040085, 0.0035332, 0.0040085, -0.0002124, 0.0002124)
6: (-0.0109595, -0.0103409, -0.0109595, -0.0103409, -0.0002764, 0.0002764)
7: (-0.0102013, -0.0101224, -0.0102013, -0.0101224, -0.0000353, 0.0000353)
8: (-0.0037094, -0.0032820, -0.0037094, -0.0032820, -0.0001910, 0.0001910)
9: (-0.0017404, 0.0003991, -0.0017404, 0.0003991, -0.0009561, 0.0009561)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.59 + 1.19 = 2.78 seconds
status: Status.ADV_EXAMPLE
