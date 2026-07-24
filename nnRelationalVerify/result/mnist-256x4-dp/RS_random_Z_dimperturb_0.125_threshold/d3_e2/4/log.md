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
Threshold: 0.0006444


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0040565, -0.0039408, -0.0040565, -0.0039408, -0.0000615, 0.0000615)
1: (0.0015683, 0.0022090, 0.0015683, 0.0022090, -0.0003408, 0.0003408)
2: (0.0100309, 0.0114625, 0.0100309, 0.0114625, -0.0007613, 0.0007613)
3: (0.0025040, 0.0031073, 0.0025040, 0.0031073, -0.0003208, 0.0003208)
4: (1.0064650, 1.0088055, 1.0064650, 1.0088055, -0.0012447, 0.0012447)
5: (0.0034991, 0.0039544, 0.0034991, 0.0039544, -0.0002421, 0.0002421)
6: (-0.0108890, -0.0102965, -0.0108890, -0.0102965, -0.0003151, 0.0003151)
7: (-0.0101924, -0.0101168, -0.0101924, -0.0101168, -0.0000402, 0.0000402)
8: (-0.0037400, -0.0033307, -0.0037400, -0.0033307, -0.0002177, 0.0002177)
9: (-0.0014969, 0.0005527, -0.0014969, 0.0005527, -0.0010899, 0.0010899)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.56 + 1.20 = 2.76 seconds
status: Status.ADV_EXAMPLE
