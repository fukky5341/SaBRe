## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.000280174


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (1.0032994, 1.0042715, 1.0032994, 1.0042715, -0.0005771, 0.0005771)
1: (-0.0004418, -0.0001996, -0.0004418, -0.0001996, -0.0001438, 0.0001438)
2: (-0.0089961, -0.0077124, -0.0089961, -0.0077124, -0.0007621, 0.0007621)
3: (0.0022372, 0.0028215, 0.0022372, 0.0028215, -0.0003469, 0.0003469)
4: (-0.0012133, -0.0009648, -0.0012133, -0.0009648, -0.0001475, 0.0001475)
5: (-0.0123553, -0.0107408, -0.0123553, -0.0107408, -0.0009585, 0.0009585)
6: (0.0042670, 0.0046768, 0.0042670, 0.0046768, -0.0002433, 0.0002433)
7: (0.0079023, 0.0089626, 0.0079023, 0.0089626, -0.0006294, 0.0006294)
8: (0.0045916, 0.0051492, 0.0045916, 0.0051492, -0.0003310, 0.0003310)
9: (-0.0078346, -0.0071880, -0.0078346, -0.0071880, -0.0003838, 0.0003838)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.29 + 1.35 = 2.64 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0002991, upper bound: 0.0002991

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 75

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002751, upper bound: 0.0002412
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002411, upper bound: 0.0002751
time: 0.55 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.23 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 1.23
Output dim: 0, lower bound: -0.0002751, upper bound: 0.0002412
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 1.23
Output dim: 0, lower bound: -0.0002411, upper bound: 0.0002751

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.64 + 1.23 = 3.86 seconds
