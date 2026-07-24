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
Threshold: 0.00046428


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041778, -0.0041457, -0.0041778, -0.0041457, -0.0000289, 0.0000289)
1: (-0.0091132, -0.0079141, -0.0091132, -0.0079141, -0.0010816, 0.0010816)
2: (0.9655272, 0.9669662, 0.9655272, 0.9669662, -0.0012980, 0.0012980)
3: (-0.0079593, 0.0026542, -0.0079593, 0.0026542, -0.0095737, 0.0095737)
4: (-0.0008949, -0.0000877, -0.0008949, -0.0000877, -0.0007281, 0.0007281)
5: (0.0163659, 0.0171817, 0.0163659, 0.0171817, -0.0007359, 0.0007359)
6: (0.0035496, 0.0039464, 0.0035496, 0.0039464, -0.0003579, 0.0003579)
7: (-0.0084661, -0.0057155, -0.0084661, -0.0057155, -0.0024811, 0.0024811)
8: (0.0100125, 0.0121947, 0.0100125, 0.0121947, -0.0019684, 0.0019684)
9: (0.0157331, 0.0196580, 0.0157331, 0.0196580, -0.0035404, 0.0035404)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.21 + 1.69 = 2.90 seconds
status: Status.ADV_EXAMPLE
