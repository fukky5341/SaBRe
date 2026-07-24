## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 6)
Time budget: 600 seconds
Split limit: 100
Threshold: 1.542703008


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.7778639, 0.6806120, -0.7778639, 0.6806120, -1.4584759, 1.4584759)
1: (-0.6040506, 0.8626654, -0.6040506, 0.8626654, -1.4667161, 1.4667161)
2: (-0.5841001, 0.8418483, -0.5841001, 0.8418483, -1.4259484, 1.4259484)
3: (-0.6007968, 0.6236554, -0.6007968, 0.6236554, -1.2244523, 1.2244523)
4: (-0.7511072, 0.7287243, -0.7511072, 0.7287243, -1.4798315, 1.4798315)
5: (-0.5852865, 1.1952124, -0.5852865, 1.1952124, -1.7804989, 1.7804989)
6: (-0.4886691, 0.6989943, -0.4886691, 0.6989943, -1.1876633, 1.1876633)
7: (-0.6124615, 0.7924250, -0.6124615, 0.7924250, -1.4048865, 1.4048865)
8: (-0.6664105, 0.8338409, -0.6664105, 0.8338409, -1.5002514, 1.5002514)
9: (-0.6932293, 0.7872544, -0.6932293, 0.7872544, -1.4804838, 1.4804838)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.63 + 3.06 = 4.70 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -1.6069825, upper bound: 1.6069825

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5428148, upper bound: 1.5428148
time: 1.52 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5428148, upper bound: 1.5428148
time: 1.47 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 3.15 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 3.15
Output dim: 5, lower bound: -1.5428148, upper bound: 1.5428148
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 3.15
Output dim: 5, lower bound: -1.5428148, upper bound: 1.5428148

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.7778639, 0.6806120, -0.7778639, 0.6806120, -1.4584759, 1.4584759
1: -0.6040506, 0.8626654, -0.6040506, 0.8626654, -1.4667161, 1.4667161
2: -0.5841001, 0.8418483, -0.5841001, 0.8418483, -1.4259484, 1.4259484
3: -0.6007968, 0.6236554, -0.6007968, 0.6236554, -1.2244523, 1.2244523
4: -0.7511072, 0.7287243, -0.7511072, 0.7287243, -1.4798315, 1.4798315
5: -0.5852865, 1.1952124, -0.5852865, 1.1952124, -1.7804989, 1.7804989
6: -0.4886691, 0.6989943, -0.4886691, 0.6989943, -1.1876633, 1.1876633
7: -0.6124615, 0.7924250, -0.6124615, 0.7924250, -1.4048865, 1.4048865
8: -0.6664105, 0.8338409, -0.6664105, 0.8338409, -1.5002514, 1.5002514
9: -0.6932293, 0.7872544, -0.6932293, 0.7872544, -1.4804838, 1.4804838

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5046753, upper bound: 1.5046753
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5046753, upper bound: 1.5046753
time: 1.34 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.7778639, 0.6806120, -0.7778639, 0.6806120, -1.4584759, 1.4584759
1: -0.6040506, 0.8626654, -0.6040506, 0.8626654, -1.4667161, 1.4667161
2: -0.5841001, 0.8418483, -0.5841001, 0.8418483, -1.4259484, 1.4259484
3: -0.6007968, 0.6236554, -0.6007968, 0.6236554, -1.2244523, 1.2244523
4: -0.7511072, 0.7287243, -0.7511072, 0.7287243, -1.4798315, 1.4798315
5: -0.5852865, 1.1952124, -0.5852865, 1.1952124, -1.7804989, 1.7804989
6: -0.4886691, 0.6989943, -0.4886691, 0.6989943, -1.1876633, 1.1876633
7: -0.6124615, 0.7924250, -0.6124615, 0.7924250, -1.4048865, 1.4048865
8: -0.6664105, 0.8338409, -0.6664105, 0.8338409, -1.5002514, 1.5002514
9: -0.6932293, 0.7872544, -0.6932293, 0.7872544, -1.4804838, 1.4804838

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5046753, upper bound: 1.5046753
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5046753, upper bound: 1.5046753
time: 1.36 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 4.30 seconds
DS_DSZ1_DSZ1, status: Status.VERIFIED, split count: 2, time: 4.30
Output dim: 5, lower bound: -1.5046753, upper bound: 1.5046753
DS_DSZ1_DSZ2, status: Status.VERIFIED, split count: 2, time: 4.30
Output dim: 5, lower bound: -1.5046753, upper bound: 1.5046753
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 4.30
Output dim: 5, lower bound: -1.5046753, upper bound: 1.5046753
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 4.30
Output dim: 5, lower bound: -1.5046753, upper bound: 1.5046753

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 4.70 + 11.76 = 16.45 seconds
