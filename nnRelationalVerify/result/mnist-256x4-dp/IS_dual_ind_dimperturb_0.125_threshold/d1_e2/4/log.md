## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 8.752e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041963, -0.0041916, -0.0041963, -0.0041916, -0.0000023, 0.0000023)
1: (-0.0098069, -0.0096298, -0.0098069, -0.0096298, -0.0000862, 0.0000862)
2: (0.9646947, 0.9649073, 0.9646947, 0.9649073, -0.0001034, 0.0001034)
3: (-0.0140998, -0.0125317, -0.0140998, -0.0125317, -0.0007626, 0.0007626)
4: (0.0002601, 0.0003793, 0.0002601, 0.0003793, -0.0000580, 0.0000580)
5: (0.0175332, 0.0176537, 0.0175332, 0.0176537, -0.0000586, 0.0000586)
6: (0.0033200, 0.0033786, 0.0033200, 0.0033786, -0.0000285, 0.0000285)
7: (-0.0045305, -0.0041242, -0.0045305, -0.0041242, -0.0001976, 0.0001976)
8: (0.0131348, 0.0134572, 0.0131348, 0.0134572, -0.0001568, 0.0001568)
9: (0.0213489, 0.0219287, 0.0213489, 0.0219287, -0.0002820, 0.0002820)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.24 + 1.18 = 2.42 seconds
status: Status.ADV_EXAMPLE
