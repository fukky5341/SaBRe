## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00086954


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0061617, 0.0085227, 0.0061617, 0.0085227, -0.0013069, 0.0013070)
1: (0.0020348, 0.0042840, 0.0020348, 0.0042840, -0.0012662, 0.0012662)
2: (-0.0201277, -0.0144986, -0.0201277, -0.0144986, -0.0028083, 0.0028083)
3: (-0.0017942, 0.0030905, -0.0017942, 0.0030905, -0.0026306, 0.0026306)
4: (0.0153887, 0.0158041, 0.0153887, 0.0158041, -0.0004076, 0.0004076)
5: (-0.0037759, 0.0031027, -0.0037759, 0.0031027, -0.0037674, 0.0037674)
6: (0.9949085, 0.9995431, 0.9949085, 0.9995431, -0.0024848, 0.0024848)
7: (0.0147757, 0.0168924, 0.0147757, 0.0168924, -0.0010406, 0.0010406)
8: (0.0039164, 0.0060017, 0.0039164, 0.0060017, -0.0012197, 0.0012197)
9: (-0.0228667, -0.0181379, -0.0228667, -0.0181379, -0.0024329, 0.0024329)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.58 + 1.27 = 2.85 seconds
status: Status.ADV_EXAMPLE
