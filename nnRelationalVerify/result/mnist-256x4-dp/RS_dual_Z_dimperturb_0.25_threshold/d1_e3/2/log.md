## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00063364


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0043315, -0.0042157, -0.0043315, -0.0042157, -0.0000847, 0.0000847)
1: (0.0030905, 0.0037314, 0.0030905, 0.0037314, -0.0004690, 0.0004690)
2: (0.0066298, 0.0080615, 0.0066298, 0.0080615, -0.0010478, 0.0010478)
3: (0.0039372, 0.0045405, 0.0039372, 0.0045405, -0.0004416, 0.0004416)
4: (1.0120251, 1.0143659, 1.0120251, 1.0143659, -0.0017131, 0.0017131)
5: (0.0045808, 0.0050361, 0.0045808, 0.0050361, -0.0003333, 0.0003333)
6: (-0.0122967, -0.0117041, -0.0122967, -0.0117041, -0.0004337, 0.0004337)
7: (-0.0103719, -0.0102963, -0.0103719, -0.0102963, -0.0000553, 0.0000553)
8: (-0.0027675, -0.0023580, -0.0027675, -0.0023580, -0.0002997, 0.0002997)
9: (-0.0063660, -0.0043162, -0.0063660, -0.0043162, -0.0015001, 0.0015001)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.27 + 1.22 = 2.49 seconds
status: Status.ADV_EXAMPLE
