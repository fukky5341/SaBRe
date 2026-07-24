## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.045187955


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0124950, 0.0040517, -0.0124950, 0.0040517, -0.0165466, 0.0165466)
1: (-0.0144604, 0.0034482, -0.0144604, 0.0034482, -0.0179086, 0.0179086)
2: (0.0184173, 0.0788744, 0.0184173, 0.0788744, -0.0604571, 0.0604571)
3: (-0.0062572, 0.0436145, -0.0062572, 0.0436145, -0.0414950, 0.0414950)
4: (-0.0088813, 0.0068967, -0.0088813, 0.0068967, -0.0157781, 0.0157781)
5: (0.0047639, 0.0204382, 0.0047639, 0.0204382, -0.0156743, 0.0156743)
6: (-0.0311039, 0.0084378, -0.0311039, 0.0084378, -0.0395418, 0.0395418)
7: (0.9000708, 0.9997994, 0.9000708, 0.9997994, -0.0997285, 0.0997285)
8: (-0.0206371, 0.0264980, -0.0206371, 0.0264980, -0.0408676, 0.0408675)
9: (-0.0266395, 0.0240074, -0.0266395, 0.0240074, -0.0506469, 0.0506469)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.36 + 1.73 = 3.09 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0517826, upper bound: 0.0517826

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0437636, upper bound: 0.0437636
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0437636, upper bound: 0.0437636
time: 0.67 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.36 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 1.36
Output dim: 7, lower bound: -0.0437636, upper bound: 0.0437636
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 1.36
Output dim: 7, lower bound: -0.0437636, upper bound: 0.0437636

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.09 + 1.36 = 4.46 seconds
