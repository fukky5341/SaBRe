## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 4.3265e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0010609, 0.0010983, 0.0010609, 0.0010983, -0.0000353, 0.0000353)
1: (0.9936450, 0.9937865, 0.9936450, 0.9937865, -0.0001160, 0.0001160)
2: (-0.0062947, -0.0055261, -0.0062947, -0.0055261, -0.0005734, 0.0005734)
3: (0.0039209, 0.0039964, 0.0039209, 0.0039964, -0.0000580, 0.0000580)
4: (0.0027846, 0.0033920, 0.0027846, 0.0033920, -0.0004689, 0.0004689)
5: (0.0062358, 0.0064019, 0.0062358, 0.0064019, -0.0001661, 0.0001661)
6: (-0.0013060, -0.0010393, -0.0013060, -0.0010393, -0.0001981, 0.0001981)
7: (-0.0082266, -0.0080813, -0.0082266, -0.0080813, -0.0001453, 0.0001453)
8: (0.0055123, 0.0065221, 0.0055123, 0.0065221, -0.0007114, 0.0007114)
9: (-0.0036825, -0.0035766, -0.0036825, -0.0035766, -0.0001059, 0.0001059)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.19 + 1.18 = 2.37 seconds
status: Status.ADV_EXAMPLE
