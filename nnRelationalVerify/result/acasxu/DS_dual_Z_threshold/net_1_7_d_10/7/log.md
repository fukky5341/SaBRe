## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_7.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 7)
Time budget: 420 seconds
Split limit: 100
Threshold: 81.860902399251


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057)
1: (-55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903)
2: (-82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490)
3: (-47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564)
4: (-75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.95 + 3.56 = 4.51 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -81.8633583, upper bound: 81.8633583

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8633184, upper bound: 81.8633184
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -81.8633184, upper bound: 81.8633583
time: 0.81 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.11 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 2.11
Output dim: 0, lower bound: -81.8633184, upper bound: 81.8633184
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 2.11
Output dim: 0, lower bound: -81.8633184, upper bound: 81.8633583

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -81.8603873, upper bound: 81.8603873
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -81.8603873, upper bound: 81.8603873
time: 0.88 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -20.8385620, 76.1094437, -20.8385620, 76.1094437, -96.9480057, 96.9480057
1: -55.3152542, 171.7260437, -55.3152542, 171.7260437, -227.0412903, 227.0412903
2: -82.9352112, 152.8937378, -82.9352112, 152.8937378, -235.8289490, 235.8289490
3: -47.5881386, 183.4647217, -47.5881386, 183.4647217, -231.0528564, 231.0528564
4: -75.8024445, 134.6393127, -75.8024445, 134.6393127, -210.4417572, 210.4417572

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -81.8603873, upper bound: 81.8603873
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -81.8603873, upper bound: 81.8603873
time: 0.82 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.58 seconds
DS_DSZ1_DSZ1, status: Status.VERIFIED, split count: 2, time: 2.58
Output dim: 0, lower bound: -81.8603873, upper bound: 81.8603873
DS_DSZ1_DSZ2, status: Status.VERIFIED, split count: 2, time: 2.58
Output dim: 0, lower bound: -81.8603873, upper bound: 81.8603873
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 2.58
Output dim: 0, lower bound: -81.8603873, upper bound: 81.8603873
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 2.58
Output dim: 0, lower bound: -81.8603873, upper bound: 81.8603873

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 4.51 + 7.38 = 11.89 seconds
