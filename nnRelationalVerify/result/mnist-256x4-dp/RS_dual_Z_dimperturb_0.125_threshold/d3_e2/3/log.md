## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00990792


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0120359, 0.0016783, -0.0120359, 0.0016783, -0.0113293, 0.0113293)
1: (-0.0122627, -0.0012493, -0.0122627, -0.0012493, -0.0106936, 0.0106936)
2: (0.0444440, 0.0506700, 0.0444440, 0.0506700, -0.0062260, 0.0062260)
3: (0.0076786, 0.0298929, 0.0076786, 0.0298929, -0.0180036, 0.0180036)
4: (-0.0041709, 0.0003739, -0.0041709, 0.0003739, -0.0045448, 0.0045448)
5: (0.0109790, 0.0136229, 0.0109790, 0.0136229, -0.0026439, 0.0026439)
6: (-0.0270638, -0.0133714, -0.0270638, -0.0133714, -0.0136924, 0.0136924)
7: (0.9178193, 0.9554207, 0.9178193, 0.9554207, -0.0376015, 0.0376015)
8: (-0.0000261, 0.0173545, -0.0000261, 0.0173545, -0.0173806, 0.0173806)
9: (-0.0082583, -0.0021551, -0.0082583, -0.0021551, -0.0061031, 0.0061031)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.56 + 1.24 = 2.80 seconds
status: Status.ADV_EXAMPLE
