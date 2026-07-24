## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00039634


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0170759, 0.0174445, 0.0170759, 0.0174445, -0.0001918, 0.0001918)
1: (-0.0004770, -0.0002271, -0.0004770, -0.0002271, -0.0001404, 0.0001404)
2: (0.0038303, 0.0039520, 0.0038303, 0.0039520, -0.0000618, 0.0000618)
3: (0.0017836, 0.0020413, 0.0017836, 0.0020413, -0.0001319, 0.0001319)
4: (-0.0040224, -0.0036484, -0.0040224, -0.0036484, -0.0001548, 0.0001548)
5: (-0.0000118, 0.0001387, -0.0000118, 0.0001387, -0.0000861, 0.0000861)
6: (-0.0037693, -0.0030349, -0.0037693, -0.0030349, -0.0003214, 0.0003214)
7: (-0.0193407, -0.0172185, -0.0193407, -0.0172185, -0.0008811, 0.0008811)
8: (0.9775810, 0.9793785, 0.9775810, 0.9793785, -0.0007629, 0.0007629)
9: (0.0036300, 0.0050013, 0.0036300, 0.0050013, -0.0005712, 0.0005712)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.32 + 1.14 = 2.46 seconds
status: Status.ADV_EXAMPLE
