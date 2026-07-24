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
Threshold: 0.00117945


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0040928, -0.0038680, -0.0040928, -0.0038680, -0.0001500, 0.0001500)
1: (0.0011649, 0.0024096, 0.0011649, 0.0024096, -0.0008303, 0.0008303)
2: (0.0095828, 0.0123636, 0.0095828, 0.0123636, -0.0018550, 0.0018550)
3: (0.0021243, 0.0032961, 0.0021243, 0.0032961, -0.0007817, 0.0007817)
4: (1.0049918, 1.0095381, 1.0049918, 1.0095381, -0.0030327, 0.0030327)
5: (0.0032125, 0.0040969, 0.0032125, 0.0040969, -0.0005900, 0.0005900)
6: (-0.0110745, -0.0099236, -0.0110745, -0.0099236, -0.0007678, 0.0007678)
7: (-0.0102160, -0.0100692, -0.0102160, -0.0100692, -0.0000979, 0.0000979)
8: (-0.0039977, -0.0032025, -0.0039977, -0.0032025, -0.0005305, 0.0005305)
9: (-0.0021383, 0.0018427, -0.0021383, 0.0018427, -0.0026556, 0.0026556)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.61 + 1.26 = 2.87 seconds
status: Status.ADV_EXAMPLE
