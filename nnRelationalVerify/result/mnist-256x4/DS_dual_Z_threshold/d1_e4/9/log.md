## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 9)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.000586925


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0033858, 0.0033858)
1: (-0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0009546, 0.0009546)
2: (-0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0070432, 0.0070432)
3: (0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0009321, 0.0009321)
4: (0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0052637, 0.0052637)
5: (0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0014624, 0.0014624)
6: (0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0013274, 0.0013274)
7: (-0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0049537, 0.0049537)
8: (-0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0038555, 0.0038555)
9: (-0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003326, 0.0003326)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.45 + 2.14 = 3.59 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0006905, upper bound: 0.0006905

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 223

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005472, upper bound: 0.0005473
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005472, upper bound: 0.0005473
time: 1.13 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.40 seconds
DS_DSZ1, status: Status.VERIFIED, split count: 1, time: 2.40
Output dim: 5, lower bound: -0.0005472, upper bound: 0.0005473
DS_DSZ2, status: Status.VERIFIED, split count: 1, time: 2.40
Output dim: 5, lower bound: -0.0005472, upper bound: 0.0005473

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.59 + 2.40 = 5.99 seconds
