## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00013584


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0039049, -0.0025483, -0.0039049, -0.0025483, -0.0013566, 0.0013566)
1: (0.0055579, 0.0064996, 0.0055579, 0.0064996, -0.0009417, 0.0009417)
2: (0.0107752, 0.0133311, 0.0107752, 0.0133311, -0.0021352, 0.0021352)
3: (-0.0040350, -0.0029010, -0.0040350, -0.0029010, -0.0011270, 0.0011270)
4: (0.0047841, 0.0051632, 0.0047841, 0.0051632, -0.0002354, 0.0002354)
5: (-0.0018483, -0.0009879, -0.0018483, -0.0009879, -0.0008604, 0.0008604)
6: (-0.0057758, -0.0053651, -0.0057758, -0.0053651, -0.0004107, 0.0004107)
7: (-0.0031336, -0.0023804, -0.0031336, -0.0023804, -0.0007532, 0.0007532)
8: (-0.0032962, -0.0016032, -0.0032962, -0.0016032, -0.0016930, 0.0016930)
9: (1.0004461, 1.0007108, 1.0004461, 1.0007108, -0.0002648, 0.0002648)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.27 + 1.23 = 2.50 seconds
status: Status.ADV_EXAMPLE
