## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 7.432e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0081712, -0.0075316, -0.0081712, -0.0075316, -0.0002930, 0.0002930)
1: (-0.0052424, -0.0050621, -0.0052424, -0.0050621, -0.0000826, 0.0000826)
2: (-0.0001199, 0.0012106, -0.0001199, 0.0012106, -0.0006095, 0.0006095)
3: (0.0016114, 0.0017875, 0.0016114, 0.0017875, -0.0000807, 0.0000807)
4: (0.0051871, 0.0061815, 0.0051871, 0.0061815, -0.0004555, 0.0004555)
5: (0.9969474, 0.9972236, 0.9969474, 0.9972236, -0.0001266, 0.0001266)
6: (0.0051128, 0.0053636, 0.0051128, 0.0053636, -0.0001149, 0.0001149)
7: (-0.0043015, -0.0033657, -0.0043015, -0.0033657, -0.0004287, 0.0004287)
8: (-0.0065734, -0.0058450, -0.0065734, -0.0058450, -0.0003337, 0.0003337)
9: (-0.0035055, -0.0034426, -0.0035055, -0.0034426, -0.0000288, 0.0000288)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.36 + 1.20 = 2.57 seconds
status: Status.ADV_EXAMPLE
