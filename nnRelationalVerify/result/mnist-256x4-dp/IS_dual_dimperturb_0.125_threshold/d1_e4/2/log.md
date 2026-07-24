## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 2.776e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0002088, 0.0006436, 0.0002088, 0.0006436, -0.0001681, 0.0001681)
1: (-0.0034481, -0.0033648, -0.0034481, -0.0033648, -0.0000270, 0.0000270)
2: (0.0151963, 0.0157458, 0.0151963, 0.0157458, -0.0002038, 0.0002038)
3: (1.0067595, 1.0069113, 1.0067595, 1.0069113, -0.0000701, 0.0000701)
4: (-0.0042114, -0.0041269, -0.0042114, -0.0041269, -0.0000300, 0.0000300)
5: (0.0041390, 0.0044712, 0.0041390, 0.0044712, -0.0001276, 0.0001276)
6: (-0.0025954, -0.0025730, -0.0025954, -0.0025730, -0.0000119, 0.0000119)
7: (-0.0126128, -0.0118088, -0.0126128, -0.0118088, -0.0003643, 0.0003643)
8: (-0.0133023, -0.0124222, -0.0133023, -0.0124222, -0.0003048, 0.0003048)
9: (0.0020274, 0.0024446, 0.0020274, 0.0024446, -0.0001398, 0.0001398)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.56 + 1.20 = 2.77 seconds
status: Status.ADV_EXAMPLE
