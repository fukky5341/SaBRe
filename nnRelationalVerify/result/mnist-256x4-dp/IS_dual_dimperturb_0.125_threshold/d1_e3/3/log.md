## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0028484


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0028023, -0.0022292, -0.0028023, -0.0022292, -0.0002355, 0.0002355)
1: (0.0237496, 0.0268565, 0.0237496, 0.0268565, -0.0012805, 0.0012805)
2: (0.0232480, 0.0253014, 0.0232480, 0.0253014, -0.0008525, 0.0008525)
3: (0.0111554, 0.0133639, 0.0111554, 0.0133639, -0.0010442, 0.0010442)
4: (-0.0135224, -0.0112389, -0.0135224, -0.0112389, -0.0010839, 0.0010839)
5: (0.0184488, 0.0210857, 0.0184488, 0.0210857, -0.0012751, 0.0012751)
6: (0.0090885, 0.0111491, 0.0090885, 0.0111491, -0.0009932, 0.0009932)
7: (-0.0183267, -0.0161666, -0.0183267, -0.0161666, -0.0009493, 0.0009493)
8: (0.0131575, 0.0152675, 0.0131575, 0.0152675, -0.0010346, 0.0010346)
9: (0.9192967, 0.9297866, 0.9192967, 0.9297866, -0.0048671, 0.0048671)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.37 + 1.17 = 2.53 seconds
status: Status.ADV_EXAMPLE
