## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 6)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0019356300000000002


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0137769, -0.0052372, -0.0137769, -0.0052372, -0.0069460, 0.0069460)
1: (-0.0068229, -0.0044152, -0.0068229, -0.0044152, -0.0019583, 0.0019583)
2: (-0.0117809, 0.0059833, -0.0117809, 0.0059833, -0.0144491, 0.0144491)
3: (0.0000683, 0.0024191, 0.0000683, 0.0024191, -0.0019121, 0.0019121)
4: (0.0016203, 0.0148962, 0.0016203, 0.0148962, -0.0107983, 0.0107983)
5: (0.9959564, 0.9996448, 0.9959564, 0.9996448, -0.0030001, 0.0030001)
6: (0.0042133, 0.0075613, 0.0042133, 0.0075613, -0.0027232, 0.0027232)
7: (-0.0076583, 0.0048358, -0.0076583, 0.0048358, -0.0101624, 0.0101624)
8: (-0.0129566, -0.0032324, -0.0129566, -0.0032324, -0.0079094, 0.0079094)
9: (-0.0037309, -0.0028919, -0.0037309, -0.0028919, -0.0006824, 0.0006824)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.12 + 3.08 = 5.21 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0021507, upper bound: 0.0021506

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020584, upper bound: 0.0020584
time: 1.87 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020584, upper bound: 0.0020584
time: 1.88 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 3.91 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 3.91
Output dim: 5, lower bound: -0.0020584, upper bound: 0.0020584
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 3.91
Output dim: 5, lower bound: -0.0020584, upper bound: 0.0020584

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137769, -0.0052372, -0.0137769, -0.0052372, -0.0069295, 0.0068962
1: -0.0068229, -0.0044152, -0.0068229, -0.0044152, -0.0019537, 0.0019443
2: -0.0117809, 0.0059833, -0.0117809, 0.0059833, -0.0144147, 0.0143454
3: 0.0000683, 0.0024191, 0.0000683, 0.0024191, -0.0019076, 0.0018984
4: 0.0016203, 0.0148962, 0.0016203, 0.0148962, -0.0107209, 0.0107727
5: 0.9959564, 0.9996448, 0.9959564, 0.9996448, -0.0029786, 0.0029930
6: 0.0042133, 0.0075613, 0.0042133, 0.0075613, -0.0027036, 0.0027167
7: -0.0076583, 0.0048358, -0.0076583, 0.0048358, -0.0100895, 0.0101383
8: -0.0129566, -0.0032324, -0.0129566, -0.0032324, -0.0078906, 0.0078527
9: -0.0037309, -0.0028919, -0.0037309, -0.0028919, -0.0006775, 0.0006808

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0016635, upper bound: 0.0016635
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0016635, upper bound: 0.0016635
time: 1.27 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137769, -0.0052372, -0.0137769, -0.0052372, -0.0068962, 0.0069460
1: -0.0068229, -0.0044152, -0.0068229, -0.0044152, -0.0019443, 0.0019583
2: -0.0117809, 0.0059833, -0.0117809, 0.0059833, -0.0143454, 0.0144491
3: 0.0000683, 0.0024191, 0.0000683, 0.0024191, -0.0018984, 0.0019121
4: 0.0016203, 0.0148962, 0.0016203, 0.0148962, -0.0107983, 0.0107209
5: 0.9959564, 0.9996448, 0.9959564, 0.9996448, -0.0030001, 0.0029786
6: 0.0042133, 0.0075613, 0.0042133, 0.0075613, -0.0027232, 0.0027036
7: -0.0076583, 0.0048358, -0.0076583, 0.0048358, -0.0101624, 0.0100895
8: -0.0129566, -0.0032324, -0.0129566, -0.0032324, -0.0078527, 0.0079094
9: -0.0037309, -0.0028919, -0.0037309, -0.0028919, -0.0006824, 0.0006775

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0016635, upper bound: 0.0016635
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0016635, upper bound: 0.0016635
time: 1.25 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 4.28 seconds
DS_DSZ1_DSZ1, status: Status.VERIFIED, split count: 2, time: 4.28
Output dim: 5, lower bound: -0.0016635, upper bound: 0.0016635
DS_DSZ1_DSZ2, status: Status.VERIFIED, split count: 2, time: 4.28
Output dim: 5, lower bound: -0.0016635, upper bound: 0.0016635
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 4.28
Output dim: 5, lower bound: -0.0016635, upper bound: 0.0016635
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 4.28
Output dim: 5, lower bound: -0.0016635, upper bound: 0.0016635

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 5.21 + 12.48 = 17.68 seconds
