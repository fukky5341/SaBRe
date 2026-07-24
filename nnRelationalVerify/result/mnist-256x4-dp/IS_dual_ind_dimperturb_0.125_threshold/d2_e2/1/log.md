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
0: (0.0030656, 0.0459753, 0.0030656, 0.0459753, -0.0285210, 0.0285210)
1: (0.0033170, 0.0042220, 0.0033170, 0.0042220, -0.0007907, 0.0007907)
2: (0.0189348, 0.0324517, 0.0189348, 0.0324517, -0.0089397, 0.0089397)
3: (0.0315659, 0.0586309, 0.0315659, 0.0586309, -0.0177036, 0.0177036)
4: (-0.0122229, -0.0053299, -0.0122229, -0.0053299, -0.0052258, 0.0052258)
5: (0.0282842, 0.0433566, 0.0282842, 0.0433566, -0.0098304, 0.0098304)
6: (-0.0043306, 0.0367532, -0.0043306, 0.0367532, -0.0268745, 0.0268745)
7: (-0.0066269, -0.0062889, -0.0066269, -0.0062889, -0.0003379, 0.0003379)
8: (0.7387683, 0.8613596, 0.7387683, 0.8613596, -0.0803356, 0.0803356)
9: (0.0764944, 0.0883928, 0.0764944, 0.0883928, -0.0079320, 0.0079320)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.32 + 1.16 = 2.49 seconds
status: Status.ADV_EXAMPLE
