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
Threshold: 0.0001407


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0083844, -0.0070102, -0.0083844, -0.0070102, -0.0008160, 0.0008160)
1: (-0.0053025, -0.0049151, -0.0053025, -0.0049151, -0.0002301, 0.0002301)
2: (-0.0005635, 0.0022952, -0.0005635, 0.0022952, -0.0016974, 0.0016974)
3: (0.0015527, 0.0019310, 0.0015527, 0.0019310, -0.0002246, 0.0002246)
4: (0.0043766, 0.0065130, 0.0043766, 0.0065130, -0.0012685, 0.0012685)
5: (0.9967222, 0.9973157, 0.9967222, 0.9973157, -0.0003524, 0.0003524)
6: (0.0049084, 0.0054471, 0.0049084, 0.0054471, -0.0003199, 0.0003199)
7: (-0.0050643, -0.0030537, -0.0050643, -0.0030537, -0.0011938, 0.0011938)
8: (-0.0068161, -0.0052513, -0.0068161, -0.0052513, -0.0009291, 0.0009291)
9: (-0.0035567, -0.0034217, -0.0035567, -0.0034217, -0.0000802, 0.0000802)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.32 + 1.29 = 2.60 seconds
status: Status.ADV_EXAMPLE
