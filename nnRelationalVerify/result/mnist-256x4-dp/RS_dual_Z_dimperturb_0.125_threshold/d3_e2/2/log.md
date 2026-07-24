## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 8.58e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041419, -0.0041357, -0.0041419, -0.0041357, -0.0000039, 0.0000039)
1: (-0.0077694, -0.0075369, -0.0077694, -0.0075369, -0.0001446, 0.0001446)
2: (0.9671398, 0.9674188, 0.9671398, 0.9674188, -0.0001735, 0.0001735)
3: (0.0039347, 0.0059931, 0.0039347, 0.0059931, -0.0012799, 0.0012799)
4: (-0.0011488, -0.0009923, -0.0011488, -0.0009923, -0.0000973, 0.0000973)
5: (0.0161092, 0.0162675, 0.0161092, 0.0162675, -0.0000984, 0.0000984)
6: (0.0039943, 0.0040713, 0.0039943, 0.0040713, -0.0000479, 0.0000479)
7: (-0.0093314, -0.0087980, -0.0093314, -0.0087980, -0.0003317, 0.0003317)
8: (0.0093260, 0.0097493, 0.0093260, 0.0097493, -0.0002631, 0.0002631)
9: (0.0144984, 0.0152596, 0.0144984, 0.0152596, -0.0004733, 0.0004733)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.64 + 1.16 = 2.79 seconds
status: Status.ADV_EXAMPLE
