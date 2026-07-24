## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_2.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 5)
Time budget: 420 seconds
Split limit: 100
Threshold: 2027.3678997182642


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-796.8801270, 1389.0648193, -796.8801270, 1389.0648193, -2185.9448242, 2185.9448242)
1: (-722.3177490, 1251.5062256, -722.3177490, 1251.5062256, -1973.8239746, 1973.8239746)
2: (-632.7144775, 1318.1571045, -632.7144775, 1318.1571045, -1950.8714600, 1950.8714600)
3: (-972.6072998, 1298.1092529, -972.6072998, 1298.1092529, -2270.7165527, 2270.7165527)
4: (-767.1251831, 1415.1643066, -767.1251831, 1415.1643066, -2182.2895508, 2182.2895508)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.58 + 2.05 = 2.63 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -2027.3881736, upper bound: 2027.3881736

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2027.3665084, upper bound: 2027.3665084
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2027.3665084, upper bound: 2027.3665084
time: 0.80 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.61 seconds
DS_DSZ1, status: Status.VERIFIED, split count: 1, time: 1.61
Output dim: 0, lower bound: -2027.3665084, upper bound: 2027.3665084
DS_DSZ2, status: Status.VERIFIED, split count: 1, time: 1.61
Output dim: 0, lower bound: -2027.3665084, upper bound: 2027.3665084

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 2.63 + 1.61 = 4.24 seconds
