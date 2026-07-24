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
0: (-0.0040612, -0.0039449, -0.0040612, -0.0039449, -0.0000507, 0.0000507)
1: (0.0015906, 0.0022345, 0.0015906, 0.0022345, -0.0002808, 0.0002808)
2: (0.0099740, 0.0114125, 0.0099740, 0.0114125, -0.0006274, 0.0006274)
3: (0.0025251, 0.0031313, 0.0025251, 0.0031313, -0.0002644, 0.0002644)
4: (1.0065467, 1.0088986, 1.0065467, 1.0088986, -0.0010257, 0.0010257)
5: (0.0035150, 0.0039725, 0.0035150, 0.0039725, -0.0001995, 0.0001995)
6: (-0.0109126, -0.0103172, -0.0109126, -0.0103172, -0.0002597, 0.0002597)
7: (-0.0101954, -0.0101194, -0.0101954, -0.0101194, -0.0000331, 0.0000331)
8: (-0.0037257, -0.0033144, -0.0037257, -0.0033144, -0.0001794, 0.0001794)
9: (-0.0015784, 0.0004810, -0.0015784, 0.0004810, -0.0008982, 0.0008982)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.61 + 1.18 = 2.79 seconds
status: Status.ADV_EXAMPLE
