## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 5.708e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041694, -0.0041618, -0.0041694, -0.0041618, -0.0000035, 0.0000035)
1: (-0.0088005, -0.0085161, -0.0088005, -0.0085161, -0.0001294, 0.0001294)
2: (0.9659024, 0.9662439, 0.9659024, 0.9662439, -0.0001553, 0.0001553)
3: (-0.0051917, -0.0026738, -0.0051917, -0.0026738, -0.0011455, 0.0011455)
4: (-0.0004897, -0.0002982, -0.0004897, -0.0002982, -0.0000871, 0.0000871)
5: (0.0167754, 0.0169690, 0.0167754, 0.0169690, -0.0000881, 0.0000881)
6: (0.0036531, 0.0037472, 0.0036531, 0.0037472, -0.0000428, 0.0000428)
7: (-0.0070853, -0.0064328, -0.0070853, -0.0064328, -0.0002969, 0.0002969)
8: (0.0111080, 0.0116257, 0.0111080, 0.0116257, -0.0002355, 0.0002355)
9: (0.0177034, 0.0186345, 0.0177034, 0.0186345, -0.0004236, 0.0004236)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.28 + 1.19 = 2.47 seconds
status: Status.ADV_EXAMPLE
