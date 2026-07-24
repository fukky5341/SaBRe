## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_8.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 4)
Time budget: 420 seconds
Split limit: 100
Threshold: 42.134888934


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.6358709, 7.2639017, -8.6358709, 7.2639017, -15.8997726, 15.8997726)
1: (-32.9253387, 26.6185799, -32.9253387, 26.6185799, -59.5439186, 59.5439186)
2: (-17.6399612, 27.3018894, -17.6399612, 27.3018894, -44.9418449, 44.9418449)
3: (-29.9307785, 24.8325806, -29.9307785, 24.8325806, -54.7633591, 54.7633591)
4: (-22.0805244, 27.8851662, -22.0805244, 27.8851662, -49.9656906, 49.9656906)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.05 + 2.33 = 3.38 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -42.1770660, upper bound: 42.1770660

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1424210, upper bound: 42.1424210
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1424210, upper bound: 42.1424210
time: 0.69 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.41 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.41
Output dim: 4, lower bound: -42.1424210, upper bound: 42.1424210
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.41
Output dim: 4, lower bound: -42.1424210, upper bound: 42.1424210

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -8.6358709, 7.2639017, -8.6358709, 7.2639017, -15.8997726, 15.8997726
1: -32.9253387, 26.6185799, -32.9253387, 26.6185799, -59.5439186, 59.5439186
2: -17.6399612, 27.3018894, -17.6399612, 27.3018894, -44.9418449, 44.9418449
3: -29.9307785, 24.8325806, -29.9307785, 24.8325806, -54.7633591, 54.7633591
4: -22.0805244, 27.8851662, -22.0805244, 27.8851662, -49.9656906, 49.9656906

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -42.1347901, upper bound: 42.1347859
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -42.1347859, upper bound: 42.1347901
time: 0.62 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -8.6358709, 7.2639017, -8.6358709, 7.2639017, -15.8997726, 15.8997726
1: -32.9253387, 26.6185799, -32.9253387, 26.6185799, -59.5439186, 59.5439186
2: -17.6399612, 27.3018894, -17.6399612, 27.3018894, -44.9418449, 44.9418449
3: -29.9307785, 24.8325806, -29.9307785, 24.8325806, -54.7633591, 54.7633591
4: -22.0805244, 27.8851662, -22.0805244, 27.8851662, -49.9656906, 49.9656906

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -42.1347901, upper bound: 42.1347859
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -42.1347859, upper bound: 42.1347901
time: 0.63 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.42 seconds
DS_DSZ1_DSZ1, status: Status.VERIFIED, split count: 2, time: 2.42
Output dim: 4, lower bound: -42.1347901, upper bound: 42.1347859
DS_DSZ1_DSZ2, status: Status.VERIFIED, split count: 2, time: 2.42
Output dim: 4, lower bound: -42.1347859, upper bound: 42.1347901
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 2.42
Output dim: 4, lower bound: -42.1347901, upper bound: 42.1347859
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 2.42
Output dim: 4, lower bound: -42.1347859, upper bound: 42.1347901

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.38 + 6.22 = 9.60 seconds
