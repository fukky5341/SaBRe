## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 5.708e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041689, -0.0041627, -0.0041689, -0.0041627, -0.0000042, 0.0000042)
1: (-0.0087822, -0.0085477, -0.0087822, -0.0085477, -0.0001565, 0.0001565)
2: (0.9659245, 0.9662058, 0.9659245, 0.9662058, -0.0001878, 0.0001878)
3: (-0.0050293, -0.0029543, -0.0050293, -0.0029543, -0.0013854, 0.0013854)
4: (-0.0004683, -0.0003105, -0.0004683, -0.0003105, -0.0001054, 0.0001054)
5: (0.0167970, 0.0169565, 0.0167970, 0.0169565, -0.0001065, 0.0001065)
6: (0.0036591, 0.0037367, 0.0036591, 0.0037367, -0.0000518, 0.0000518)
7: (-0.0070126, -0.0064749, -0.0070126, -0.0064749, -0.0003590, 0.0003590)
8: (0.0111657, 0.0115923, 0.0111657, 0.0115923, -0.0002848, 0.0002848)
9: (0.0178071, 0.0185745, 0.0178071, 0.0185745, -0.0005123, 0.0005123)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.38 + 1.19 = 2.56 seconds
status: Status.ADV_EXAMPLE
