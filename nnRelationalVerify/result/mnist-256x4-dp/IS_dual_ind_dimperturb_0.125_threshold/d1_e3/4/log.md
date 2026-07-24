## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00039634


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0170966, 0.0174737, 0.0170966, 0.0174737, -0.0001910, 0.0001910)
1: (-0.0004691, -0.0002074, -0.0004691, -0.0002074, -0.0001381, 0.0001381)
2: (0.0038211, 0.0039439, 0.0038211, 0.0039439, -0.0000615, 0.0000615)
3: (0.0017863, 0.0020658, 0.0017863, 0.0020658, -0.0001284, 0.0001284)
4: (-0.0040334, -0.0037008, -0.0040334, -0.0037008, -0.0001421, 0.0001421)
5: (-0.0000223, 0.0001347, -0.0000223, 0.0001347, -0.0000840, 0.0000840)
6: (-0.0037069, -0.0029834, -0.0037069, -0.0029834, -0.0003037, 0.0003037)
7: (-0.0194026, -0.0175171, -0.0194026, -0.0175171, -0.0008102, 0.0008102)
8: (0.9775215, 0.9791273, 0.9775215, 0.9791273, -0.0007195, 0.0007195)
9: (0.0038234, 0.0050407, 0.0038234, 0.0050407, -0.0005269, 0.0005269)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.41 + 1.16 = 2.57 seconds
status: Status.ADV_EXAMPLE
