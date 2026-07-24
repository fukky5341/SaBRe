## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 6.64e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0040690, -0.0040648, -0.0040690, -0.0040648, -0.0000019, 0.0000019)
1: (-0.0050391, -0.0048810, -0.0050391, -0.0048810, -0.0000718, 0.0000718)
2: (0.9704163, 0.9706060, 0.9704163, 0.9706060, -0.0000862, 0.0000862)
3: (0.0281019, 0.0295008, 0.0281019, 0.0295008, -0.0006357, 0.0006357)
4: (-0.0029367, -0.0028303, -0.0029367, -0.0028303, -0.0000483, 0.0000483)
5: (0.0143022, 0.0144098, 0.0143022, 0.0144098, -0.0000489, 0.0000489)
6: (0.0048979, 0.0049502, 0.0048979, 0.0049502, -0.0000238, 0.0000238)
7: (-0.0154237, -0.0150611, -0.0154237, -0.0150611, -0.0001647, 0.0001647)
8: (0.0044927, 0.0047804, 0.0044927, 0.0047804, -0.0001307, 0.0001307)
9: (0.0058053, 0.0063226, 0.0058053, 0.0063226, -0.0002351, 0.0002351)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.19 + 1.13 = 2.32 seconds
status: Status.ADV_EXAMPLE
