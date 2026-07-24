## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.09377412512571143


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099)
1: (-0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061)
2: (-0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713)
3: (-0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883)
4: (-0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926)
5: (-0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117)
6: (-0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370)
7: (-0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255)
8: (-0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314)
9: (0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.32 + 7.94 = 9.25 seconds
status: Status.VERIFIED
relational distance
Output dim: 9, lower bound: -0.0842349, upper bound: 0.0842349
