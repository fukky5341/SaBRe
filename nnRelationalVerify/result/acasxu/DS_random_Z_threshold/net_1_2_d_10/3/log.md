## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_2.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 3)
Time budget: 420 seconds
Split limit: 100
Threshold: 0.5653432899999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970)
1: (-0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661)
2: (-0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567)
3: (-0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351)
4: (-0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.79 + 0.93 = 1.71 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.5950982, upper bound: 0.5950982

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5193791, upper bound: 0.5193791
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5193791, upper bound: 0.5193791
time: 0.26 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.55 seconds
DS_DSZ1, status: Status.VERIFIED, split count: 1, time: 0.55
Output dim: 0, lower bound: -0.5193791, upper bound: 0.5193791
DS_DSZ2, status: Status.VERIFIED, split count: 1, time: 0.55
Output dim: 0, lower bound: -0.5193791, upper bound: 0.5193791

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 1.71 + 0.55 = 2.26 seconds
