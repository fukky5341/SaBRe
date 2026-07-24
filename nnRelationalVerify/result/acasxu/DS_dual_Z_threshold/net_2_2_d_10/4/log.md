## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_2.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 4)
Time budget: 420 seconds
Split limit: 100
Threshold: 339.41632513516


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-149.8139038, 226.3787994, -149.8139038, 226.3787994, -376.1926880, 376.1926880)
1: (-118.9682159, 218.5465088, -118.9682159, 218.5465088, -337.5147095, 337.5147095)
2: (-102.7016296, 224.2848206, -102.7016296, 224.2848206, -326.9864502, 326.9864502)
3: (-155.0733032, 221.6566010, -155.0733032, 221.6566010, -376.7298584, 376.7298584)
4: (-123.6448669, 237.1756439, -123.6448669, 237.1756439, -360.8204956, 360.8204956)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.46 + 2.04 = 3.50 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -339.4332968, upper bound: 339.4332968

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4185114, upper bound: 339.4185114
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -339.4185114, upper bound: 339.4206701
time: 0.96 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.16 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 2.16
Output dim: 4, lower bound: -339.4185114, upper bound: 339.4185114
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 2.16
Output dim: 4, lower bound: -339.4185114, upper bound: 339.4206701

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -149.8139038, 226.3787994, -149.8139038, 226.3787994, -376.1926880, 376.1926880
1: -118.9682159, 218.5465088, -118.9682159, 218.5465088, -337.5147095, 337.5147095
2: -102.7016296, 224.2848206, -102.7016296, 224.2848206, -326.9864502, 326.9864502
3: -155.0733032, 221.6566010, -155.0733032, 221.6566010, -376.7298584, 376.7298584
4: -123.6448669, 237.1756439, -123.6448669, 237.1756439, -360.8204956, 360.8204956

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4129027, upper bound: 339.4129027
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4135168, upper bound: 339.4129027
time: 1.04 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -149.8139038, 226.3787994, -149.8139038, 226.3787994, -376.1926880, 376.1926880
1: -118.9682159, 218.5465088, -118.9682159, 218.5465088, -337.5147095, 337.5147095
2: -102.7016296, 224.2848206, -102.7016296, 224.2848206, -326.9864502, 326.9864502
3: -155.0733032, 221.6566010, -155.0733032, 221.6566010, -376.7298584, 376.7298584
4: -123.6448669, 237.1756439, -123.6448669, 237.1756439, -360.8204956, 360.8204956

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4129027, upper bound: 339.4135168
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -339.4129027, upper bound: 339.4135168
time: 0.94 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.30 seconds
DS_DSZ1_DSZ1, status: Status.VERIFIED, split count: 2, time: 3.30
Output dim: 4, lower bound: -339.4129027, upper bound: 339.4129027
DS_DSZ1_DSZ2, status: Status.VERIFIED, split count: 2, time: 3.30
Output dim: 4, lower bound: -339.4135168, upper bound: 339.4129027
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 3.30
Output dim: 4, lower bound: -339.4129027, upper bound: 339.4135168
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 3.30
Output dim: 4, lower bound: -339.4129027, upper bound: 339.4135168

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.50 + 8.88 = 12.39 seconds
