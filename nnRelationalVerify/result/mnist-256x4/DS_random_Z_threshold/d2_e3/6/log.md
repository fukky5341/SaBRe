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
execution time: IAR + RelationalAnalysis = 0.76 + 2.84 = 3.60 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0021507, upper bound: 0.0021506

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 171

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020818, upper bound: 0.0020817
time: 1.74 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020818, upper bound: 0.0020818
time: 1.74 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 3.50 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 3.50
Output dim: 5, lower bound: -0.0020818, upper bound: 0.0020817
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 3.50
Output dim: 5, lower bound: -0.0020818, upper bound: 0.0020818

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137769, -0.0052372, -0.0137769, -0.0052372, -0.0069149, 0.0069005
1: -0.0068229, -0.0044152, -0.0068229, -0.0044152, -0.0019496, 0.0019455
2: -0.0117809, 0.0059833, -0.0117809, 0.0059833, -0.0143843, 0.0143544
3: 0.0000683, 0.0024191, 0.0000683, 0.0024191, -0.0019035, 0.0018996
4: 0.0016203, 0.0148962, 0.0016203, 0.0148962, -0.0107276, 0.0107500
5: 0.9959564, 0.9996448, 0.9959564, 0.9996448, -0.0029804, 0.0029867
6: 0.0042133, 0.0075613, 0.0042133, 0.0075613, -0.0027053, 0.0027110
7: -0.0076583, 0.0048358, -0.0076583, 0.0048358, -0.0100959, 0.0101169
8: -0.0129566, -0.0032324, -0.0129566, -0.0032324, -0.0078740, 0.0078576
9: -0.0037309, -0.0028919, -0.0037309, -0.0028919, -0.0006779, 0.0006793

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020485, upper bound: 0.0020485
time: 1.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020485, upper bound: 0.0020485
time: 1.77 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137769, -0.0052372, -0.0137769, -0.0052372, -0.0069005, 0.0069460
1: -0.0068229, -0.0044152, -0.0068229, -0.0044152, -0.0019455, 0.0019583
2: -0.0117809, 0.0059833, -0.0117809, 0.0059833, -0.0143544, 0.0144491
3: 0.0000683, 0.0024191, 0.0000683, 0.0024191, -0.0018996, 0.0019121
4: 0.0016203, 0.0148962, 0.0016203, 0.0148962, -0.0107983, 0.0107276
5: 0.9959564, 0.9996448, 0.9959564, 0.9996448, -0.0030001, 0.0029804
6: 0.0042133, 0.0075613, 0.0042133, 0.0075613, -0.0027232, 0.0027053
7: -0.0076583, 0.0048358, -0.0076583, 0.0048358, -0.0101624, 0.0100959
8: -0.0129566, -0.0032324, -0.0129566, -0.0032324, -0.0078576, 0.0079094
9: -0.0037309, -0.0028919, -0.0037309, -0.0028919, -0.0006824, 0.0006779

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020404, upper bound: 0.0020374
time: 1.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020374, upper bound: 0.0020403
time: 1.84 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.93 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.93
Output dim: 5, lower bound: -0.0020485, upper bound: 0.0020485
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.93
Output dim: 5, lower bound: -0.0020485, upper bound: 0.0020485
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.93
Output dim: 5, lower bound: -0.0020404, upper bound: 0.0020374
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.93
Output dim: 5, lower bound: -0.0020374, upper bound: 0.0020403

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137769, -0.0052372, -0.0137769, -0.0052372, -0.0068165, 0.0068050
1: -0.0068229, -0.0044152, -0.0068229, -0.0044152, -0.0019218, 0.0019186
2: -0.0117809, 0.0059833, -0.0117809, 0.0059833, -0.0141796, 0.0141558
3: 0.0000683, 0.0024191, 0.0000683, 0.0024191, -0.0018764, 0.0018733
4: 0.0016203, 0.0148962, 0.0016203, 0.0148962, -0.0105791, 0.0105970
5: 0.9959564, 0.9996448, 0.9959564, 0.9996448, -0.0029392, 0.0029442
6: 0.0042133, 0.0075613, 0.0042133, 0.0075613, -0.0026679, 0.0026724
7: -0.0076583, 0.0048358, -0.0076583, 0.0048358, -0.0099561, 0.0099729
8: -0.0129566, -0.0032324, -0.0129566, -0.0032324, -0.0077619, 0.0077489
9: -0.0037309, -0.0028919, -0.0037309, -0.0028919, -0.0006685, 0.0006697

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019902, upper bound: 0.0019861
time: 1.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019861, upper bound: 0.0019902
time: 1.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137769, -0.0052372, -0.0137769, -0.0052372, -0.0068194, 0.0067951
1: -0.0068229, -0.0044152, -0.0068229, -0.0044152, -0.0019226, 0.0019158
2: -0.0117809, 0.0059833, -0.0117809, 0.0059833, -0.0141857, 0.0141351
3: 0.0000683, 0.0024191, 0.0000683, 0.0024191, -0.0018772, 0.0018706
4: 0.0016203, 0.0148962, 0.0016203, 0.0148962, -0.0105637, 0.0106015
5: 0.9959564, 0.9996448, 0.9959564, 0.9996448, -0.0029349, 0.0029454
6: 0.0042133, 0.0075613, 0.0042133, 0.0075613, -0.0026640, 0.0026735
7: -0.0076583, 0.0048358, -0.0076583, 0.0048358, -0.0099416, 0.0099772
8: -0.0129566, -0.0032324, -0.0129566, -0.0032324, -0.0077653, 0.0077376
9: -0.0037309, -0.0028919, -0.0037309, -0.0028919, -0.0006676, 0.0006699

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019063, upper bound: 0.0019063
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019063, upper bound: 0.0019063
time: 1.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137769, -0.0052372, -0.0137769, -0.0052372, -0.0067249, 0.0067888
1: -0.0068229, -0.0044152, -0.0068229, -0.0044152, -0.0018960, 0.0019140
2: -0.0117809, 0.0059833, -0.0117809, 0.0059833, -0.0139892, 0.0141221
3: 0.0000683, 0.0024191, 0.0000683, 0.0024191, -0.0018512, 0.0018688
4: 0.0016203, 0.0148962, 0.0016203, 0.0148962, -0.0105540, 0.0104547
5: 0.9959564, 0.9996448, 0.9959564, 0.9996448, -0.0029322, 0.0029046
6: 0.0042133, 0.0075613, 0.0042133, 0.0075613, -0.0026616, 0.0026365
7: -0.0076583, 0.0048358, -0.0076583, 0.0048358, -0.0099325, 0.0098390
8: -0.0129566, -0.0032324, -0.0129566, -0.0032324, -0.0076577, 0.0077305
9: -0.0037309, -0.0028919, -0.0037309, -0.0028919, -0.0006669, 0.0006607

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 174

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020138, upper bound: 0.0020071
time: 1.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020101, upper bound: 0.0020108
time: 1.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137769, -0.0052372, -0.0137769, -0.0052372, -0.0067437, 0.0067700
1: -0.0068229, -0.0044152, -0.0068229, -0.0044152, -0.0019013, 0.0019087
2: -0.0117809, 0.0059833, -0.0117809, 0.0059833, -0.0140283, 0.0140830
3: 0.0000683, 0.0024191, 0.0000683, 0.0024191, -0.0018564, 0.0018637
4: 0.0016203, 0.0148962, 0.0016203, 0.0148962, -0.0105247, 0.0104839
5: 0.9959564, 0.9996448, 0.9959564, 0.9996448, -0.0029241, 0.0029127
6: 0.0042133, 0.0075613, 0.0042133, 0.0075613, -0.0026542, 0.0026439
7: -0.0076583, 0.0048358, -0.0076583, 0.0048358, -0.0099049, 0.0098665
8: -0.0129566, -0.0032324, -0.0129566, -0.0032324, -0.0076791, 0.0077090
9: -0.0037309, -0.0028919, -0.0037309, -0.0028919, -0.0006651, 0.0006625

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018655, upper bound: 0.0018724
time: 1.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018655, upper bound: 0.0018724
time: 1.54 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.90 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.90
Output dim: 5, lower bound: -0.0019902, upper bound: 0.0019861
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.90
Output dim: 5, lower bound: -0.0019861, upper bound: 0.0019902
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 3.90
Output dim: 5, lower bound: -0.0019063, upper bound: 0.0019063
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 3.90
Output dim: 5, lower bound: -0.0019063, upper bound: 0.0019063
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.90
Output dim: 5, lower bound: -0.0020138, upper bound: 0.0020071
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.90
Output dim: 5, lower bound: -0.0020101, upper bound: 0.0020108
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 3.90
Output dim: 5, lower bound: -0.0018655, upper bound: 0.0018724
DS_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 3.90
Output dim: 5, lower bound: -0.0018655, upper bound: 0.0018724

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137769, -0.0052372, -0.0137769, -0.0052372, -0.0067563, 0.0067669
1: -0.0068229, -0.0044152, -0.0068229, -0.0044152, -0.0019049, 0.0019079
2: -0.0117809, 0.0059833, -0.0117809, 0.0059833, -0.0140545, 0.0140766
3: 0.0000683, 0.0024191, 0.0000683, 0.0024191, -0.0018599, 0.0018628
4: 0.0016203, 0.0148962, 0.0016203, 0.0148962, -0.0105200, 0.0105035
5: 0.9959564, 0.9996448, 0.9959564, 0.9996448, -0.0029228, 0.0029182
6: 0.0042133, 0.0075613, 0.0042133, 0.0075613, -0.0026530, 0.0026488
7: -0.0076583, 0.0048358, -0.0076583, 0.0048358, -0.0099005, 0.0098849
8: -0.0129566, -0.0032324, -0.0129566, -0.0032324, -0.0076935, 0.0077056
9: -0.0037309, -0.0028919, -0.0037309, -0.0028919, -0.0006648, 0.0006638

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 101

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019239, upper bound: 0.0019188
time: 2.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019223, upper bound: 0.0019202
time: 2.02 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137769, -0.0052372, -0.0137769, -0.0052372, -0.0067697, 0.0067449
1: -0.0068229, -0.0044152, -0.0068229, -0.0044152, -0.0019086, 0.0019016
2: -0.0117809, 0.0059833, -0.0117809, 0.0059833, -0.0140823, 0.0140307
3: 0.0000683, 0.0024191, 0.0000683, 0.0024191, -0.0018636, 0.0018567
4: 0.0016203, 0.0148962, 0.0016203, 0.0148962, -0.0104856, 0.0105242
5: 0.9959564, 0.9996448, 0.9959564, 0.9996448, -0.0029132, 0.0029239
6: 0.0042133, 0.0075613, 0.0042133, 0.0075613, -0.0026443, 0.0026541
7: -0.0076583, 0.0048358, -0.0076583, 0.0048358, -0.0098682, 0.0099045
8: -0.0129566, -0.0032324, -0.0129566, -0.0032324, -0.0077087, 0.0076804
9: -0.0037309, -0.0028919, -0.0037309, -0.0028919, -0.0006626, 0.0006651

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019820, upper bound: 0.0019781
time: 2.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019747, upper bound: 0.0019860
time: 1.94 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137769, -0.0052372, -0.0137769, -0.0052372, -0.0066747, 0.0067510
1: -0.0068229, -0.0044152, -0.0068229, -0.0044152, -0.0018818, 0.0019034
2: -0.0117809, 0.0059833, -0.0117809, 0.0059833, -0.0138847, 0.0140435
3: 0.0000683, 0.0024191, 0.0000683, 0.0024191, -0.0018374, 0.0018584
4: 0.0016203, 0.0148962, 0.0016203, 0.0148962, -0.0104953, 0.0103765
5: 0.9959564, 0.9996448, 0.9959564, 0.9996448, -0.0029159, 0.0028829
6: 0.0042133, 0.0075613, 0.0042133, 0.0075613, -0.0026468, 0.0026168
7: -0.0076583, 0.0048358, -0.0076583, 0.0048358, -0.0098772, 0.0097655
8: -0.0129566, -0.0032324, -0.0129566, -0.0032324, -0.0076005, 0.0076875
9: -0.0037309, -0.0028919, -0.0037309, -0.0028919, -0.0006632, 0.0006557

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019715, upper bound: 0.0019601
time: 1.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019656, upper bound: 0.0019665
time: 1.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137769, -0.0052372, -0.0137769, -0.0052372, -0.0066846, 0.0067386
1: -0.0068229, -0.0044152, -0.0068229, -0.0044152, -0.0018846, 0.0018999
2: -0.0117809, 0.0059833, -0.0117809, 0.0059833, -0.0139054, 0.0140176
3: 0.0000683, 0.0024191, 0.0000683, 0.0024191, -0.0018402, 0.0018550
4: 0.0016203, 0.0148962, 0.0016203, 0.0148962, -0.0104759, 0.0103920
5: 0.9959564, 0.9996448, 0.9959564, 0.9996448, -0.0029105, 0.0028872
6: 0.0042133, 0.0075613, 0.0042133, 0.0075613, -0.0026419, 0.0026207
7: -0.0076583, 0.0048358, -0.0076583, 0.0048358, -0.0098590, 0.0097800
8: -0.0129566, -0.0032324, -0.0129566, -0.0032324, -0.0076118, 0.0076732
9: -0.0037309, -0.0028919, -0.0037309, -0.0028919, -0.0006620, 0.0006567

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 165

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019331, upper bound: 0.0019319
time: 1.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019331, upper bound: 0.0019316
time: 1.89 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 4.44 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 4.44
Output dim: 5, lower bound: -0.0019239, upper bound: 0.0019188
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 4.44
Output dim: 5, lower bound: -0.0019223, upper bound: 0.0019202
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 5, lower bound: -0.0019820, upper bound: 0.0019781
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 5, lower bound: -0.0019747, upper bound: 0.0019860
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 5, lower bound: -0.0019715, upper bound: 0.0019601
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 5, lower bound: -0.0019656, upper bound: 0.0019665
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 4.44
Output dim: 5, lower bound: -0.0019331, upper bound: 0.0019319
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 4.44
Output dim: 5, lower bound: -0.0019331, upper bound: 0.0019316

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137769, -0.0052372, -0.0137769, -0.0052372, -0.0067790, 0.0067612
1: -0.0068229, -0.0044152, -0.0068229, -0.0044152, -0.0019112, 0.0019062
2: -0.0117809, 0.0059833, -0.0117809, 0.0059833, -0.0141016, 0.0140647
3: 0.0000683, 0.0024191, 0.0000683, 0.0024191, -0.0018661, 0.0018612
4: 0.0016203, 0.0148962, 0.0016203, 0.0148962, -0.0105111, 0.0105387
5: 0.9959564, 0.9996448, 0.9959564, 0.9996448, -0.0029203, 0.0029280
6: 0.0042133, 0.0075613, 0.0042133, 0.0075613, -0.0026507, 0.0026577
7: -0.0076583, 0.0048358, -0.0076583, 0.0048358, -0.0098921, 0.0099181
8: -0.0129566, -0.0032324, -0.0129566, -0.0032324, -0.0077193, 0.0076990
9: -0.0037309, -0.0028919, -0.0037309, -0.0028919, -0.0006642, 0.0006660

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019458, upper bound: 0.0019416
time: 2.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019458, upper bound: 0.0019416
time: 1.96 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137769, -0.0052372, -0.0137769, -0.0052372, -0.0067853, 0.0067541
1: -0.0068229, -0.0044152, -0.0068229, -0.0044152, -0.0019130, 0.0019042
2: -0.0117809, 0.0059833, -0.0117809, 0.0059833, -0.0141149, 0.0140500
3: 0.0000683, 0.0024191, 0.0000683, 0.0024191, -0.0018679, 0.0018593
4: 0.0016203, 0.0148962, 0.0016203, 0.0148962, -0.0105001, 0.0105486
5: 0.9959564, 0.9996448, 0.9959564, 0.9996448, -0.0029172, 0.0029307
6: 0.0042133, 0.0075613, 0.0042133, 0.0075613, -0.0026480, 0.0026602
7: -0.0076583, 0.0048358, -0.0076583, 0.0048358, -0.0098817, 0.0099274
8: -0.0129566, -0.0032324, -0.0129566, -0.0032324, -0.0077265, 0.0076910
9: -0.0037309, -0.0028919, -0.0037309, -0.0028919, -0.0006635, 0.0006666

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018272, upper bound: 0.0018383
time: 1.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018272, upper bound: 0.0018383
time: 1.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137769, -0.0052372, -0.0137769, -0.0052372, -0.0066133, 0.0067122
1: -0.0068229, -0.0044152, -0.0068229, -0.0044152, -0.0018645, 0.0018924
2: -0.0117809, 0.0059833, -0.0117809, 0.0059833, -0.0137569, 0.0139628
3: 0.0000683, 0.0024191, 0.0000683, 0.0024191, -0.0018205, 0.0018477
4: 0.0016203, 0.0148962, 0.0016203, 0.0148962, -0.0104349, 0.0102811
5: 0.9959564, 0.9996448, 0.9959564, 0.9996448, -0.0028991, 0.0028564
6: 0.0042133, 0.0075613, 0.0042133, 0.0075613, -0.0026315, 0.0025927
7: -0.0076583, 0.0048358, -0.0076583, 0.0048358, -0.0098204, 0.0096756
8: -0.0129566, -0.0032324, -0.0129566, -0.0032324, -0.0075306, 0.0076432
9: -0.0037309, -0.0028919, -0.0037309, -0.0028919, -0.0006594, 0.0006497

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 216

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019663, upper bound: 0.0019496
time: 1.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019592, upper bound: 0.0019541
time: 1.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137769, -0.0052372, -0.0137769, -0.0052372, -0.0066347, 0.0066896
1: -0.0068229, -0.0044152, -0.0068229, -0.0044152, -0.0018706, 0.0018861
2: -0.0117809, 0.0059833, -0.0117809, 0.0059833, -0.0138016, 0.0139158
3: 0.0000683, 0.0024191, 0.0000683, 0.0024191, -0.0018264, 0.0018415
4: 0.0016203, 0.0148962, 0.0016203, 0.0148962, -0.0103998, 0.0103145
5: 0.9959564, 0.9996448, 0.9959564, 0.9996448, -0.0028894, 0.0028657
6: 0.0042133, 0.0075613, 0.0042133, 0.0075613, -0.0026227, 0.0026012
7: -0.0076583, 0.0048358, -0.0076583, 0.0048358, -0.0097874, 0.0097070
8: -0.0129566, -0.0032324, -0.0129566, -0.0032324, -0.0075550, 0.0076175
9: -0.0037309, -0.0028919, -0.0037309, -0.0028919, -0.0006572, 0.0006518

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 216

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018004, upper bound: 0.0017992
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018004, upper bound: 0.0017992
time: 1.60 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 5.72 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.72
Output dim: 5, lower bound: -0.0019458, upper bound: 0.0019416
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.72
Output dim: 5, lower bound: -0.0019458, upper bound: 0.0019416
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.72
Output dim: 5, lower bound: -0.0018272, upper bound: 0.0018383
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.72
Output dim: 5, lower bound: -0.0018272, upper bound: 0.0018383
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.72
Output dim: 5, lower bound: -0.0019663, upper bound: 0.0019496
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.72
Output dim: 5, lower bound: -0.0019592, upper bound: 0.0019541
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.72
Output dim: 5, lower bound: -0.0018004, upper bound: 0.0017992
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.72
Output dim: 5, lower bound: -0.0018004, upper bound: 0.0017992

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137769, -0.0052372, -0.0137769, -0.0052372, -0.0067497, 0.0067307
1: -0.0068229, -0.0044152, -0.0068229, -0.0044152, -0.0019030, 0.0018976
2: -0.0117809, 0.0059833, -0.0117809, 0.0059833, -0.0140408, 0.0140012
3: 0.0000683, 0.0024191, 0.0000683, 0.0024191, -0.0018581, 0.0018528
4: 0.0016203, 0.0148962, 0.0016203, 0.0148962, -0.0104636, 0.0104932
5: 0.9959564, 0.9996448, 0.9959564, 0.9996448, -0.0029071, 0.0029153
6: 0.0042133, 0.0075613, 0.0042133, 0.0075613, -0.0026388, 0.0026462
7: -0.0076583, 0.0048358, -0.0076583, 0.0048358, -0.0098474, 0.0098753
8: -0.0129566, -0.0032324, -0.0129566, -0.0032324, -0.0076860, 0.0076643
9: -0.0037309, -0.0028919, -0.0037309, -0.0028919, -0.0006612, 0.0006631

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0016734, upper bound: 0.0016698
time: 1.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0016734, upper bound: 0.0016698
time: 1.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137769, -0.0052372, -0.0137769, -0.0052372, -0.0067484, 0.0067287
1: -0.0068229, -0.0044152, -0.0068229, -0.0044152, -0.0019026, 0.0018971
2: -0.0117809, 0.0059833, -0.0117809, 0.0059833, -0.0140381, 0.0139971
3: 0.0000683, 0.0024191, 0.0000683, 0.0024191, -0.0018577, 0.0018523
4: 0.0016203, 0.0148962, 0.0016203, 0.0148962, -0.0104606, 0.0104912
5: 0.9959564, 0.9996448, 0.9959564, 0.9996448, -0.0029063, 0.0029148
6: 0.0042133, 0.0075613, 0.0042133, 0.0075613, -0.0026380, 0.0026457
7: -0.0076583, 0.0048358, -0.0076583, 0.0048358, -0.0098446, 0.0098734
8: -0.0129566, -0.0032324, -0.0129566, -0.0032324, -0.0076845, 0.0076620
9: -0.0037309, -0.0028919, -0.0037309, -0.0028919, -0.0006610, 0.0006630

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017966, upper bound: 0.0017898
time: 1.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017966, upper bound: 0.0017898
time: 1.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137769, -0.0052372, -0.0137769, -0.0052372, -0.0066149, 0.0067227
1: -0.0068229, -0.0044152, -0.0068229, -0.0044152, -0.0018650, 0.0018954
2: -0.0117809, 0.0059833, -0.0117809, 0.0059833, -0.0137604, 0.0139846
3: 0.0000683, 0.0024191, 0.0000683, 0.0024191, -0.0018210, 0.0018506
4: 0.0016203, 0.0148962, 0.0016203, 0.0148962, -0.0104512, 0.0102837
5: 0.9959564, 0.9996448, 0.9959564, 0.9996448, -0.0029037, 0.0028571
6: 0.0042133, 0.0075613, 0.0042133, 0.0075613, -0.0026356, 0.0025934
7: -0.0076583, 0.0048358, -0.0076583, 0.0048358, -0.0098358, 0.0096781
8: -0.0129566, -0.0032324, -0.0129566, -0.0032324, -0.0075325, 0.0076552
9: -0.0037309, -0.0028919, -0.0037309, -0.0028919, -0.0006605, 0.0006499

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019266, upper bound: 0.0019109
time: 1.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019266, upper bound: 0.0019107
time: 2.04 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137769, -0.0052372, -0.0137769, -0.0052372, -0.0066209, 0.0067147
1: -0.0068229, -0.0044152, -0.0068229, -0.0044152, -0.0018667, 0.0018931
2: -0.0117809, 0.0059833, -0.0117809, 0.0059833, -0.0137728, 0.0139679
3: 0.0000683, 0.0024191, 0.0000683, 0.0024191, -0.0018226, 0.0018484
4: 0.0016203, 0.0148962, 0.0016203, 0.0148962, -0.0104388, 0.0102929
5: 0.9959564, 0.9996448, 0.9959564, 0.9996448, -0.0029002, 0.0028597
6: 0.0042133, 0.0075613, 0.0042133, 0.0075613, -0.0026325, 0.0025957
7: -0.0076583, 0.0048358, -0.0076583, 0.0048358, -0.0098240, 0.0096868
8: -0.0129566, -0.0032324, -0.0129566, -0.0032324, -0.0075392, 0.0076461
9: -0.0037309, -0.0028919, -0.0037309, -0.0028919, -0.0006597, 0.0006504

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019566, upper bound: 0.0019535
time: 1.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019586, upper bound: 0.0019524
time: 1.69 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 4.32 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.32
Output dim: 5, lower bound: -0.0016734, upper bound: 0.0016698
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.32
Output dim: 5, lower bound: -0.0016734, upper bound: 0.0016698
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.32
Output dim: 5, lower bound: -0.0017966, upper bound: 0.0017898
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.32
Output dim: 5, lower bound: -0.0017966, upper bound: 0.0017898
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.32
Output dim: 5, lower bound: -0.0019266, upper bound: 0.0019109
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.32
Output dim: 5, lower bound: -0.0019266, upper bound: 0.0019107
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 5, lower bound: -0.0019566, upper bound: 0.0019535
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.32
Output dim: 5, lower bound: -0.0019586, upper bound: 0.0019524

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137769, -0.0052372, -0.0137769, -0.0052372, -0.0066043, 0.0066887
1: -0.0068229, -0.0044152, -0.0068229, -0.0044152, -0.0018620, 0.0018858
2: -0.0117809, 0.0059833, -0.0117809, 0.0059833, -0.0137384, 0.0139138
3: 0.0000683, 0.0024191, 0.0000683, 0.0024191, -0.0018181, 0.0018413
4: 0.0016203, 0.0148962, 0.0016203, 0.0148962, -0.0103983, 0.0102672
5: 0.9959564, 0.9996448, 0.9959564, 0.9996448, -0.0028890, 0.0028525
6: 0.0042133, 0.0075613, 0.0042133, 0.0075613, -0.0026223, 0.0025892
7: -0.0076583, 0.0048358, -0.0076583, 0.0048358, -0.0097859, 0.0096626
8: -0.0129566, -0.0032324, -0.0129566, -0.0032324, -0.0075204, 0.0076164
9: -0.0037309, -0.0028919, -0.0037309, -0.0028919, -0.0006571, 0.0006488

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017945, upper bound: 0.0017915
time: 1.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017945, upper bound: 0.0017915
time: 1.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137769, -0.0052372, -0.0137769, -0.0052372, -0.0065960, 0.0066976
1: -0.0068229, -0.0044152, -0.0068229, -0.0044152, -0.0018597, 0.0018883
2: -0.0117809, 0.0059833, -0.0117809, 0.0059833, -0.0137210, 0.0139324
3: 0.0000683, 0.0024191, 0.0000683, 0.0024191, -0.0018158, 0.0018437
4: 0.0016203, 0.0148962, 0.0016203, 0.0148962, -0.0104122, 0.0102542
5: 0.9959564, 0.9996448, 0.9959564, 0.9996448, -0.0028928, 0.0028489
6: 0.0042133, 0.0075613, 0.0042133, 0.0075613, -0.0026258, 0.0025860
7: -0.0076583, 0.0048358, -0.0076583, 0.0048358, -0.0097991, 0.0096504
8: -0.0129566, -0.0032324, -0.0129566, -0.0032324, -0.0075109, 0.0076266
9: -0.0037309, -0.0028919, -0.0037309, -0.0028919, -0.0006580, 0.0006480

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 198

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019514, upper bound: 0.0019319
time: 2.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019367, upper bound: 0.0019451
time: 1.93 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 4.68 seconds
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.68
Output dim: 5, lower bound: -0.0017945, upper bound: 0.0017915
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.68
Output dim: 5, lower bound: -0.0017945, upper bound: 0.0017915
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.68
Output dim: 5, lower bound: -0.0019514, upper bound: 0.0019319
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.68
Output dim: 5, lower bound: -0.0019367, upper bound: 0.0019451

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0137769, -0.0052372, -0.0137769, -0.0052372, -0.0065596, 0.0066785
1: -0.0068229, -0.0044152, -0.0068229, -0.0044152, -0.0018494, 0.0018829
2: -0.0117809, 0.0059833, -0.0117809, 0.0059833, -0.0136453, 0.0138927
3: 0.0000683, 0.0024191, 0.0000683, 0.0024191, -0.0018057, 0.0018385
4: 0.0016203, 0.0148962, 0.0016203, 0.0148962, -0.0103825, 0.0101976
5: 0.9959564, 0.9996448, 0.9959564, 0.9996448, -0.0028846, 0.0028332
6: 0.0042133, 0.0075613, 0.0042133, 0.0075613, -0.0026183, 0.0025717
7: -0.0076583, 0.0048358, -0.0076583, 0.0048358, -0.0097711, 0.0095971
8: -0.0129566, -0.0032324, -0.0129566, -0.0032324, -0.0074694, 0.0076049
9: -0.0037309, -0.0028919, -0.0037309, -0.0028919, -0.0006561, 0.0006444

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018911, upper bound: 0.0018878
time: 2.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018911, upper bound: 0.0018878
time: 2.36 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0137769, -0.0052372, -0.0137769, -0.0052372, -0.0065773, 0.0066587
1: -0.0068229, -0.0044152, -0.0068229, -0.0044152, -0.0018544, 0.0018773
2: -0.0117809, 0.0059833, -0.0117809, 0.0059833, -0.0136821, 0.0138514
3: 0.0000683, 0.0024191, 0.0000683, 0.0024191, -0.0018106, 0.0018330
4: 0.0016203, 0.0148962, 0.0016203, 0.0148962, -0.0103517, 0.0102252
5: 0.9959564, 0.9996448, 0.9959564, 0.9996448, -0.0028760, 0.0028409
6: 0.0042133, 0.0075613, 0.0042133, 0.0075613, -0.0026105, 0.0025786
7: -0.0076583, 0.0048358, -0.0076583, 0.0048358, -0.0097421, 0.0096230
8: -0.0129566, -0.0032324, -0.0129566, -0.0032324, -0.0074896, 0.0075823
9: -0.0037309, -0.0028919, -0.0037309, -0.0028919, -0.0006542, 0.0006462

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 101

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018777, upper bound: 0.0018817
time: 1.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018772, upper bound: 0.0018837
time: 1.89 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 4.59 seconds
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 5, lower bound: -0.0018911, upper bound: 0.0018878
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 5, lower bound: -0.0018911, upper bound: 0.0018878
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 5, lower bound: -0.0018777, upper bound: 0.0018817
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 5, lower bound: -0.0018772, upper bound: 0.0018837

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.60 + 106.89 = 110.49 seconds
