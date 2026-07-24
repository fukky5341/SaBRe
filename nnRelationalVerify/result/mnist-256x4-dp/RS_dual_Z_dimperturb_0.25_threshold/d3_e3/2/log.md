## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.002363328


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0001921, 0.0011309, 0.0001921, 0.0011309, -0.0009388, 0.0009388)
1: (0.9932348, 0.9958084, 0.9932348, 0.9958084, -0.0025737, 0.0025737)
2: (-0.0106124, -0.0020091, -0.0106124, -0.0020091, -0.0080521, 0.0080521)
3: (0.0029284, 0.0042822, 0.0029284, 0.0042822, -0.0013538, 0.0013538)
4: (0.0000049, 0.0068045, 0.0000049, 0.0068045, -0.0067995, 0.0067995)
5: (0.0038175, 0.0071620, 0.0038175, 0.0071620, -0.0033445, 0.0033445)
6: (-0.0028047, 0.0002970, -0.0028047, 0.0002970, -0.0031018, 0.0031018)
7: (-0.0088916, -0.0067270, -0.0088916, -0.0067270, -0.0021646, 0.0021646)
8: (0.0008914, 0.0121950, 0.0008914, 0.0121950, -0.0111535, 0.0111535)
9: (-0.0037185, -0.0010105, -0.0037185, -0.0010105, -0.0027080, 0.0027080)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.69 + 1.70 = 3.39 seconds
status: Status.VERIFIED
relational distance
Output dim: 1, lower bound: -0.0022184, upper bound: 0.0022184
