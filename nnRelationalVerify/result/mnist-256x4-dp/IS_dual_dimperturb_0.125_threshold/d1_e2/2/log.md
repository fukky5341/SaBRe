## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 6.64e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0040688, -0.0040643, -0.0040688, -0.0040643, -0.0000021, 0.0000021)
1: (-0.0050322, -0.0048650, -0.0050322, -0.0048650, -0.0000778, 0.0000778)
2: (0.9704245, 0.9706253, 0.9704245, 0.9706253, -0.0000933, 0.0000933)
3: (0.0281626, 0.0296429, 0.0281626, 0.0296429, -0.0006883, 0.0006883)
4: (-0.0029475, -0.0028350, -0.0029475, -0.0028350, -0.0000523, 0.0000523)
5: (0.0142913, 0.0144051, 0.0142913, 0.0144051, -0.0000529, 0.0000529)
6: (0.0049001, 0.0049555, 0.0049001, 0.0049555, -0.0000257, 0.0000257)
7: (-0.0154605, -0.0150768, -0.0154605, -0.0150768, -0.0001784, 0.0001784)
8: (0.0044635, 0.0047679, 0.0044635, 0.0047679, -0.0001415, 0.0001415)
9: (0.0057527, 0.0063001, 0.0057527, 0.0063001, -0.0002545, 0.0002545)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.29 + 1.17 = 2.46 seconds
status: Status.ADV_EXAMPLE
