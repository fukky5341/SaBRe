## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00100386


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0021384, -0.0014960, -0.0021384, -0.0014960, -0.0002929, 0.0002929)
1: (-0.0023726, -0.0006183, -0.0023726, -0.0006183, -0.0008359, 0.0008359)
2: (0.0046566, 0.0061398, 0.0046566, 0.0061398, -0.0007258, 0.0007258)
3: (-0.0041621, -0.0040094, -0.0041621, -0.0040094, -0.0000818, 0.0000818)
4: (0.0045551, 0.0058103, 0.0045551, 0.0058103, -0.0006215, 0.0006215)
5: (-0.0010031, 0.0002935, -0.0010031, 0.0002935, -0.0006616, 0.0006616)
6: (-0.0055699, -0.0048756, -0.0055699, -0.0048756, -0.0003319, 0.0003319)
7: (0.0008995, 0.0020678, 0.0008995, 0.0020678, -0.0005591, 0.0005591)
8: (-0.0004042, -0.0002404, -0.0004042, -0.0002404, -0.0000956, 0.0000956)
9: (1.0049703, 1.0080550, 1.0049703, 1.0080550, -0.0015626, 0.0015626)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.42 + 1.21 = 2.64 seconds
status: Status.VERIFIED
relational distance
Output dim: 9, lower bound: -0.0009420, upper bound: 0.0009420
