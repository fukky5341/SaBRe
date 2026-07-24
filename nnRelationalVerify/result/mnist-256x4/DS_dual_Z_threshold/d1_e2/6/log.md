## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 6)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.01169311


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0056372, 0.0058689, -0.0056372, 0.0058689, -0.0112354, 0.0112354)
1: (0.0015671, 0.0125474, 0.0015671, 0.0125474, -0.0109804, 0.0109804)
2: (0.0139266, 0.0312011, 0.0139266, 0.0312011, -0.0151616, 0.0151616)
3: (-0.0100901, -0.0020237, -0.0100901, -0.0020237, -0.0080664, 0.0080664)
4: (-0.0022926, 0.0028996, -0.0022926, 0.0028996, -0.0051922, 0.0051922)
5: (-0.0042298, 0.0073507, -0.0042298, 0.0073507, -0.0115806, 0.0115806)
6: (-0.0070884, -0.0006062, -0.0070884, -0.0006062, -0.0064822, 0.0064822)
7: (-0.0114622, 0.0008301, -0.0114622, 0.0008301, -0.0122923, 0.0122923)
8: (-0.0113532, 0.0008013, -0.0113532, 0.0008013, -0.0121544, 0.0121544)
9: (0.9805366, 0.9990062, 0.9805366, 0.9990062, -0.0184696, 0.0184696)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.21 + 1.90 = 3.12 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0137566, upper bound: 0.0137566

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135016, upper bound: 0.0134847
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0134847, upper bound: 0.0135016
time: 0.99 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.12 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 2.12
Output dim: 9, lower bound: -0.0135016, upper bound: 0.0134847
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 2.12
Output dim: 9, lower bound: -0.0134847, upper bound: 0.0135016

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0056372, 0.0058689, -0.0056372, 0.0058689, -0.0112349, 0.0112347
1: 0.0015671, 0.0125474, 0.0015671, 0.0125474, -0.0109804, 0.0109804
2: 0.0139266, 0.0312011, 0.0139266, 0.0312011, -0.0151552, 0.0151528
3: -0.0100901, -0.0020237, -0.0100901, -0.0020237, -0.0080664, 0.0080664
4: -0.0022926, 0.0028996, -0.0022926, 0.0028996, -0.0051922, 0.0051922
5: -0.0042298, 0.0073507, -0.0042298, 0.0073507, -0.0115806, 0.0115806
6: -0.0070884, -0.0006062, -0.0070884, -0.0006062, -0.0064822, 0.0064822
7: -0.0114622, 0.0008301, -0.0114622, 0.0008301, -0.0122923, 0.0122923
8: -0.0113532, 0.0008013, -0.0113532, 0.0008013, -0.0121544, 0.0121544
9: 0.9805366, 0.9990062, 0.9805366, 0.9990062, -0.0184696, 0.0184696

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0110360, upper bound: 0.0110360
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0110360, upper bound: 0.0110360
time: 0.75 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0056372, 0.0058689, -0.0056372, 0.0058689, -0.0112354, 0.0112349
1: 0.0015671, 0.0125474, 0.0015671, 0.0125474, -0.0109804, 0.0109804
2: 0.0139266, 0.0312011, 0.0139266, 0.0312011, -0.0151528, 0.0151616
3: -0.0100901, -0.0020237, -0.0100901, -0.0020237, -0.0080664, 0.0080664
4: -0.0022926, 0.0028996, -0.0022926, 0.0028996, -0.0051922, 0.0051922
5: -0.0042298, 0.0073507, -0.0042298, 0.0073507, -0.0115806, 0.0115806
6: -0.0070884, -0.0006062, -0.0070884, -0.0006062, -0.0064822, 0.0064822
7: -0.0114622, 0.0008301, -0.0114622, 0.0008301, -0.0122923, 0.0122923
8: -0.0113532, 0.0008013, -0.0113532, 0.0008013, -0.0121544, 0.0121544
9: 0.9805366, 0.9990062, 0.9805366, 0.9990062, -0.0184696, 0.0184696

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0110360, upper bound: 0.0110360
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0110360, upper bound: 0.0110360
time: 0.75 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.63 seconds
DS_DSZ1_DSZ1, status: Status.VERIFIED, split count: 2, time: 2.63
Output dim: 9, lower bound: -0.0110360, upper bound: 0.0110360
DS_DSZ1_DSZ2, status: Status.VERIFIED, split count: 2, time: 2.63
Output dim: 9, lower bound: -0.0110360, upper bound: 0.0110360
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 2.63
Output dim: 9, lower bound: -0.0110360, upper bound: 0.0110360
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 2.63
Output dim: 9, lower bound: -0.0110360, upper bound: 0.0110360

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.12 + 7.38 = 10.50 seconds
