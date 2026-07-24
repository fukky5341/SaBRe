## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.03610424


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0031797, 0.0497615, 0.0031797, 0.0497615, -0.0298792, 0.0298792)
1: (0.0033178, 0.0043959, 0.0033178, 0.0043959, -0.0009514, 0.0009514)
2: (0.0189686, 0.0337652, 0.0189686, 0.0337652, -0.0095163, 0.0095163)
3: (0.0316362, 0.0611184, 0.0316362, 0.0611184, -0.0187067, 0.0187067)
4: (-0.0122046, -0.0047217, -0.0122046, -0.0047217, -0.0055742, 0.0055742)
5: (0.0283240, 0.0447037, 0.0283240, 0.0447037, -0.0103309, 0.0103309)
6: (-0.0042245, 0.0405299, -0.0042245, 0.0405299, -0.0283979, 0.0283979)
7: (-0.0066567, -0.0062657, -0.0066567, -0.0062657, -0.0003910, 0.0003910)
8: (0.7279511, 0.8610337, 0.7279511, 0.8610337, -0.0841942, 0.0841942)
9: (0.0765261, 0.0894426, 0.0765261, 0.0894426, -0.0083078, 0.0083078)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.32 + 1.15 = 2.47 seconds
status: Status.ADV_EXAMPLE
