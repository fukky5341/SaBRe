## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 10)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0059612499999999995


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974)
1: (-0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539)
2: (0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646)
3: (-0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001)
4: (-0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409)
5: (0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188)
6: (-0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569)
7: (0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987)
8: (-0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740)
9: (-0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.47 + 2.94 = 4.40 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0062749, upper bound: 0.0062749

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062602, upper bound: 0.0062598
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062598, upper bound: 0.0062602
time: 1.33 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.88 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 2.88
Output dim: 7, lower bound: -0.0062602, upper bound: 0.0062598
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 2.88
Output dim: 7, lower bound: -0.0062598, upper bound: 0.0062602

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062370, upper bound: 0.0062369
time: 1.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062370, upper bound: 0.0062433
time: 1.90 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062433, upper bound: 0.0062370
time: 1.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062370, upper bound: 0.0062436
time: 1.40 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 4.16 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.16
Output dim: 7, lower bound: -0.0062370, upper bound: 0.0062369
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.16
Output dim: 7, lower bound: -0.0062370, upper bound: 0.0062433
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.16
Output dim: 7, lower bound: -0.0062433, upper bound: 0.0062370
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.16
Output dim: 7, lower bound: -0.0062370, upper bound: 0.0062436

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062343, upper bound: 0.0062357
time: 1.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062424, upper bound: 0.0062342
time: 1.35 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062343, upper bound: 0.0062422
time: 1.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062358, upper bound: 0.0062395
time: 1.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062395, upper bound: 0.0062358
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062422, upper bound: 0.0062344
time: 1.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062342, upper bound: 0.0062424
time: 1.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062357, upper bound: 0.0062401
time: 1.30 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 4.07 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 7, lower bound: -0.0062343, upper bound: 0.0062357
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 7, lower bound: -0.0062424, upper bound: 0.0062342
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 7, lower bound: -0.0062343, upper bound: 0.0062422
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 7, lower bound: -0.0062358, upper bound: 0.0062395
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 7, lower bound: -0.0062395, upper bound: 0.0062358
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 7, lower bound: -0.0062422, upper bound: 0.0062344
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 7, lower bound: -0.0062342, upper bound: 0.0062424
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 7, lower bound: -0.0062357, upper bound: 0.0062401

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062231, upper bound: 0.0062242
time: 2.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062231, upper bound: 0.0062242
time: 2.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062231, upper bound: 0.0062231
time: 1.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062311, upper bound: 0.0062231
time: 1.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062231, upper bound: 0.0062303
time: 1.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062236, upper bound: 0.0062302
time: 2.12 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062231, upper bound: 0.0062282
time: 2.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062248, upper bound: 0.0062282
time: 1.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062231, upper bound: 0.0062248
time: 2.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062231, upper bound: 0.0062248
time: 1.91 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062231, upper bound: 0.0062236
time: 1.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062304, upper bound: 0.0062236
time: 1.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062231, upper bound: 0.0062311
time: 1.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062231, upper bound: 0.0062311
time: 1.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062231, upper bound: 0.0062291
time: 2.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062231, upper bound: 0.0062290
time: 1.42 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 4.99 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.99
Output dim: 7, lower bound: -0.0062231, upper bound: 0.0062242
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.99
Output dim: 7, lower bound: -0.0062231, upper bound: 0.0062242
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.99
Output dim: 7, lower bound: -0.0062231, upper bound: 0.0062231
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.99
Output dim: 7, lower bound: -0.0062311, upper bound: 0.0062231
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.99
Output dim: 7, lower bound: -0.0062231, upper bound: 0.0062303
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.99
Output dim: 7, lower bound: -0.0062236, upper bound: 0.0062302
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.99
Output dim: 7, lower bound: -0.0062231, upper bound: 0.0062282
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.99
Output dim: 7, lower bound: -0.0062248, upper bound: 0.0062282
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.99
Output dim: 7, lower bound: -0.0062231, upper bound: 0.0062248
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.99
Output dim: 7, lower bound: -0.0062231, upper bound: 0.0062248
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.99
Output dim: 7, lower bound: -0.0062231, upper bound: 0.0062236
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.99
Output dim: 7, lower bound: -0.0062304, upper bound: 0.0062236
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.99
Output dim: 7, lower bound: -0.0062231, upper bound: 0.0062311
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.99
Output dim: 7, lower bound: -0.0062231, upper bound: 0.0062311
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.99
Output dim: 7, lower bound: -0.0062231, upper bound: 0.0062291
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.99
Output dim: 7, lower bound: -0.0062231, upper bound: 0.0062290

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 127

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062143, upper bound: 0.0062172
time: 2.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062215, upper bound: 0.0062157
time: 1.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 127

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062210, upper bound: 0.0062172
time: 1.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062143, upper bound: 0.0062157
time: 2.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 127

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062143, upper bound: 0.0062162
time: 2.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062236, upper bound: 0.0062143
time: 1.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 127

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062143, upper bound: 0.0062161
time: 2.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062236, upper bound: 0.0062143
time: 2.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 127

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062143, upper bound: 0.0062230
time: 1.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062165, upper bound: 0.0062218
time: 2.01 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 127

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062144, upper bound: 0.0062230
time: 2.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062143, upper bound: 0.0062215
time: 2.03 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 127

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062143, upper bound: 0.0062209
time: 2.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062176, upper bound: 0.0062197
time: 1.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 127

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062167, upper bound: 0.0062209
time: 1.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062176, upper bound: 0.0062197
time: 1.90 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 127

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062143, upper bound: 0.0062176
time: 1.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062143, upper bound: 0.0062167
time: 1.95 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 127

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062143, upper bound: 0.0062176
time: 1.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062143, upper bound: 0.0062167
time: 2.00 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 127

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062143, upper bound: 0.0062165
time: 1.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062229, upper bound: 0.0062155
time: 1.97 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 127

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062143, upper bound: 0.0062165
time: 2.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062144, upper bound: 0.0062154
time: 1.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 127

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062143, upper bound: 0.0062236
time: 2.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062143, upper bound: 0.0062230
time: 1.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 127

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062143, upper bound: 0.0062236
time: 1.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062143, upper bound: 0.0062230
time: 2.03 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 127

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062157, upper bound: 0.0062215
time: 1.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062143, upper bound: 0.0062210
time: 1.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 127

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062157, upper bound: 0.0062215
time: 2.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062144, upper bound: 0.0062210
time: 1.96 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 5.48 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.48
Output dim: 7, lower bound: -0.0062143, upper bound: 0.0062172
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.48
Output dim: 7, lower bound: -0.0062215, upper bound: 0.0062157
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.48
Output dim: 7, lower bound: -0.0062210, upper bound: 0.0062172
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.48
Output dim: 7, lower bound: -0.0062143, upper bound: 0.0062157
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.48
Output dim: 7, lower bound: -0.0062143, upper bound: 0.0062162
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.48
Output dim: 7, lower bound: -0.0062236, upper bound: 0.0062143
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.48
Output dim: 7, lower bound: -0.0062143, upper bound: 0.0062161
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.48
Output dim: 7, lower bound: -0.0062236, upper bound: 0.0062143
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.48
Output dim: 7, lower bound: -0.0062143, upper bound: 0.0062230
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.48
Output dim: 7, lower bound: -0.0062165, upper bound: 0.0062218
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.48
Output dim: 7, lower bound: -0.0062144, upper bound: 0.0062230
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.48
Output dim: 7, lower bound: -0.0062143, upper bound: 0.0062215
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.48
Output dim: 7, lower bound: -0.0062143, upper bound: 0.0062209
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.48
Output dim: 7, lower bound: -0.0062176, upper bound: 0.0062197
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.48
Output dim: 7, lower bound: -0.0062167, upper bound: 0.0062209
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.48
Output dim: 7, lower bound: -0.0062176, upper bound: 0.0062197
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.48
Output dim: 7, lower bound: -0.0062143, upper bound: 0.0062176
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.48
Output dim: 7, lower bound: -0.0062143, upper bound: 0.0062167
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.48
Output dim: 7, lower bound: -0.0062143, upper bound: 0.0062176
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.48
Output dim: 7, lower bound: -0.0062143, upper bound: 0.0062167
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.48
Output dim: 7, lower bound: -0.0062143, upper bound: 0.0062165
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.48
Output dim: 7, lower bound: -0.0062229, upper bound: 0.0062155
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.48
Output dim: 7, lower bound: -0.0062143, upper bound: 0.0062165
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.48
Output dim: 7, lower bound: -0.0062144, upper bound: 0.0062154
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.48
Output dim: 7, lower bound: -0.0062143, upper bound: 0.0062236
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.48
Output dim: 7, lower bound: -0.0062143, upper bound: 0.0062230
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.48
Output dim: 7, lower bound: -0.0062143, upper bound: 0.0062236
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.48
Output dim: 7, lower bound: -0.0062143, upper bound: 0.0062230
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.48
Output dim: 7, lower bound: -0.0062157, upper bound: 0.0062215
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.48
Output dim: 7, lower bound: -0.0062143, upper bound: 0.0062210
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.48
Output dim: 7, lower bound: -0.0062157, upper bound: 0.0062215
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.48
Output dim: 7, lower bound: -0.0062144, upper bound: 0.0062210

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062132, upper bound: 0.0062076
time: 1.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062107, upper bound: 0.0062092
time: 1.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062064
time: 1.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062111, upper bound: 0.0062076
time: 2.01 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062132, upper bound: 0.0062076
time: 2.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062092
time: 1.94 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062139, upper bound: 0.0062063
time: 1.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062111, upper bound: 0.0062076
time: 1.93 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062151, upper bound: 0.0062059
time: 1.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062135, upper bound: 0.0062084
time: 1.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062157, upper bound: 0.0062046
time: 2.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062064
time: 1.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062150, upper bound: 0.0062059
time: 1.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062135, upper bound: 0.0062084
time: 2.03 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062047
time: 2.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062064
time: 2.03 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062077, upper bound: 0.0062133
time: 2.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062055, upper bound: 0.0062152
time: 1.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062125
time: 1.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062137
time: 2.03 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062077, upper bound: 0.0062131
time: 1.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062055, upper bound: 0.0062150
time: 2.03 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062087, upper bound: 0.0062121
time: 1.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062062, upper bound: 0.0062135
time: 1.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062086, upper bound: 0.0062107
time: 1.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062133
time: 1.94 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062095, upper bound: 0.0062097
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062120
time: 1.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062086, upper bound: 0.0062107
time: 1.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062073, upper bound: 0.0062133
time: 2.02 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062095, upper bound: 0.0062096
time: 1.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062120
time: 1.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062120, upper bound: 0.0062081
time: 1.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062096, upper bound: 0.0062095
time: 2.12 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062133, upper bound: 0.0062073
time: 1.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062086
time: 1.50 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062120, upper bound: 0.0062081
time: 1.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062095
time: 1.91 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062073
time: 1.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062107, upper bound: 0.0062086
time: 2.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062062
time: 2.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062121, upper bound: 0.0062087
time: 2.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062150, upper bound: 0.0062055
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062077
time: 1.98 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062138, upper bound: 0.0062062
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062125, upper bound: 0.0062087
time: 1.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062151, upper bound: 0.0062055
time: 1.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062133, upper bound: 0.0062077
time: 1.85 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062064, upper bound: 0.0062140
time: 2.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062046, upper bound: 0.0062157
time: 1.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062135
time: 1.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062150
time: 1.98 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062064, upper bound: 0.0062140
time: 1.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062157
time: 2.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062135
time: 1.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062150
time: 1.89 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062111
time: 2.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062139
time: 1.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062107
time: 2.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062132
time: 2.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062111
time: 1.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062140
time: 2.12 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062107
time: 1.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062076, upper bound: 0.0062132
time: 1.86 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 5.29 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062132, upper bound: 0.0062076
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062107, upper bound: 0.0062092
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062064
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062111, upper bound: 0.0062076
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062132, upper bound: 0.0062076
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062092
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062139, upper bound: 0.0062063
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062111, upper bound: 0.0062076
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062151, upper bound: 0.0062059
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062135, upper bound: 0.0062084
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062157, upper bound: 0.0062046
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062064
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062150, upper bound: 0.0062059
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062135, upper bound: 0.0062084
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062047
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062064
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062077, upper bound: 0.0062133
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062055, upper bound: 0.0062152
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062125
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062137
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062077, upper bound: 0.0062131
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062055, upper bound: 0.0062150
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062087, upper bound: 0.0062121
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062062, upper bound: 0.0062135
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062086, upper bound: 0.0062107
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062133
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062095, upper bound: 0.0062097
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062120
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062086, upper bound: 0.0062107
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062073, upper bound: 0.0062133
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062095, upper bound: 0.0062096
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062120
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062120, upper bound: 0.0062081
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062096, upper bound: 0.0062095
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062133, upper bound: 0.0062073
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062086
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062120, upper bound: 0.0062081
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062095
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062073
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062107, upper bound: 0.0062086
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062062
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062121, upper bound: 0.0062087
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062150, upper bound: 0.0062055
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062077
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062138, upper bound: 0.0062062
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062125, upper bound: 0.0062087
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062151, upper bound: 0.0062055
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062133, upper bound: 0.0062077
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062064, upper bound: 0.0062140
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062046, upper bound: 0.0062157
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062135
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062150
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062064, upper bound: 0.0062140
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062157
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062135
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062150
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062111
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062139
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062107
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062132
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062111
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062140
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062107
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.29
Output dim: 7, lower bound: -0.0062076, upper bound: 0.0062132

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061951, upper bound: 0.0061823
time: 1.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061883, upper bound: 0.0061900
time: 1.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061927, upper bound: 0.0061831
time: 1.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061869, upper bound: 0.0061915
time: 1.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061957, upper bound: 0.0061809
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061892, upper bound: 0.0061885
time: 1.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061932, upper bound: 0.0061817
time: 1.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061878, upper bound: 0.0061897
time: 1.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061951, upper bound: 0.0061824
time: 1.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061883, upper bound: 0.0061901
time: 1.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061927, upper bound: 0.0061831
time: 1.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061868, upper bound: 0.0061915
time: 2.03 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061957, upper bound: 0.0061809
time: 1.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061890, upper bound: 0.0061884
time: 1.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061932, upper bound: 0.0061817
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061877, upper bound: 0.0061897
time: 1.98 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061966, upper bound: 0.0061813
time: 1.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061890, upper bound: 0.0061884
time: 1.89 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061952, upper bound: 0.0061824
time: 1.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061800, upper bound: 0.0061907
time: 1.97 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061974, upper bound: 0.0061799
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061900, upper bound: 0.0061870
time: 1.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061960, upper bound: 0.0061808
time: 1.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061893, upper bound: 0.0061887
time: 1.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061966, upper bound: 0.0061813
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061889, upper bound: 0.0061884
time: 1.87 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061952, upper bound: 0.0061824
time: 1.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061883, upper bound: 0.0061906
time: 1.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061974, upper bound: 0.0061799
time: 1.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061899, upper bound: 0.0061869
time: 1.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061960, upper bound: 0.0061808
time: 1.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061892, upper bound: 0.0061887
time: 1.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061900, upper bound: 0.0061889
time: 1.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061816, upper bound: 0.0061952
time: 2.12 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061879, upper bound: 0.0061897
time: 1.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061805, upper bound: 0.0061968
time: 1.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061911, upper bound: 0.0061874
time: 1.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061830, upper bound: 0.0061941
time: 1.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061888, upper bound: 0.0061882
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061800, upper bound: 0.0061953
time: 1.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061900, upper bound: 0.0061889
time: 1.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061814, upper bound: 0.0061951
time: 1.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061879, upper bound: 0.0061897
time: 1.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061803, upper bound: 0.0061967
time: 1.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061911, upper bound: 0.0061874
time: 1.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061826, upper bound: 0.0061939
time: 2.02 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061888, upper bound: 0.0061882
time: 1.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061800, upper bound: 0.0061952
time: 1.97 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061907, upper bound: 0.0061873
time: 1.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061823, upper bound: 0.0061927
time: 2.05 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061895, upper bound: 0.0061887
time: 1.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061815, upper bound: 0.0061951
time: 1.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061800, upper bound: 0.0061861
time: 2.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061835, upper bound: 0.0061918
time: 2.08 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061906, upper bound: 0.0061875
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061827, upper bound: 0.0061938
time: 2.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061907, upper bound: 0.0061873
time: 1.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061800, upper bound: 0.0061927
time: 2.08 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061895, upper bound: 0.0061887
time: 1.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061813, upper bound: 0.0061951
time: 1.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061919, upper bound: 0.0061861
time: 1.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061833, upper bound: 0.0061917
time: 1.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061906, upper bound: 0.0061875
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061824, upper bound: 0.0061938
time: 1.55 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061800, upper bound: 0.0061824
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061800, upper bound: 0.0061906
time: 2.07 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061800, upper bound: 0.0061833
time: 2.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061800, upper bound: 0.0061919
time: 2.03 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061951, upper bound: 0.0061813
time: 1.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061887, upper bound: 0.0061895
time: 2.11 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061927, upper bound: 0.0061821
time: 2.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061800, upper bound: 0.0061907
time: 1.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061800, upper bound: 0.0061827
time: 1.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061800, upper bound: 0.0061906
time: 2.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061800, upper bound: 0.0061835
time: 1.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061800, upper bound: 0.0061919
time: 2.07 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061951, upper bound: 0.0061815
time: 1.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061887, upper bound: 0.0061895
time: 2.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061927, upper bound: 0.0061823
time: 2.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061800, upper bound: 0.0061907
time: 2.14 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061952, upper bound: 0.0061816
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061800, upper bound: 0.0061888
time: 1.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061939, upper bound: 0.0061826
time: 2.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061873, upper bound: 0.0061910
time: 1.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061968, upper bound: 0.0061804
time: 1.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061897, upper bound: 0.0061879
time: 2.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061951, upper bound: 0.0061814
time: 1.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061889, upper bound: 0.0061900
time: 2.06 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061953, upper bound: 0.0061819
time: 1.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061882, upper bound: 0.0061888
time: 1.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061941, upper bound: 0.0061830
time: 1.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061874, upper bound: 0.0061910
time: 1.49 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061968, upper bound: 0.0061806
time: 1.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061897, upper bound: 0.0061879
time: 2.00 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061952, upper bound: 0.0061816
time: 1.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061888, upper bound: 0.0061900
time: 1.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061800, upper bound: 0.0061893
time: 2.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061800, upper bound: 0.0061959
time: 1.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061800, upper bound: 0.0061899
time: 1.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061799, upper bound: 0.0061974
time: 1.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061906, upper bound: 0.0061883
time: 2.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061824, upper bound: 0.0061953
time: 1.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061884, upper bound: 0.0061890
time: 1.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061800, upper bound: 0.0061966
time: 1.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061800, upper bound: 0.0061893
time: 1.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061800, upper bound: 0.0061960
time: 2.08 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061870, upper bound: 0.0061900
time: 1.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061799, upper bound: 0.0061974
time: 2.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0040164, 0.0010811, -0.0040164, 0.0010811, -0.0050974, 0.0050974
1: -0.0060763, -0.0020223, -0.0060763, -0.0020223, -0.0040539, 0.0040539
2: 0.0304678, 0.0373324, 0.0304678, 0.0373324, -0.0068646, 0.0068646
3: -0.0031577, 0.0016423, -0.0031577, 0.0016423, -0.0048001, 0.0048001
4: -0.0050736, 0.0010673, -0.0050736, 0.0010673, -0.0061409, 0.0061409
5: 0.0090636, 0.0143824, 0.0090636, 0.0143824, -0.0053188, 0.0053188
6: -0.0078564, 0.0019005, -0.0078564, 0.0019005, -0.0097569, 0.0097569
7: 0.9710903, 0.9793890, 0.9710903, 0.9793890, -0.0082987, 0.0082987
8: -0.0167479, -0.0010739, -0.0167479, -0.0010739, -0.0156740, 0.0156740
9: -0.0033814, 0.0057947, -0.0033814, 0.0057947, -0.0091761, 0.0091761

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061907, upper bound: 0.0061884
time: 1.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061824, upper bound: 0.0061953
time: 1.85 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 4.95 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061951, upper bound: 0.0061823
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061883, upper bound: 0.0061900
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061927, upper bound: 0.0061831
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061869, upper bound: 0.0061915
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061957, upper bound: 0.0061809
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061892, upper bound: 0.0061885
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061932, upper bound: 0.0061817
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061878, upper bound: 0.0061897
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061951, upper bound: 0.0061824
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061883, upper bound: 0.0061901
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061927, upper bound: 0.0061831
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061868, upper bound: 0.0061915
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061957, upper bound: 0.0061809
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061890, upper bound: 0.0061884
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061932, upper bound: 0.0061817
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061877, upper bound: 0.0061897
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061966, upper bound: 0.0061813
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061890, upper bound: 0.0061884
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061952, upper bound: 0.0061824
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061800, upper bound: 0.0061907
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061974, upper bound: 0.0061799
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061900, upper bound: 0.0061870
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061960, upper bound: 0.0061808
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061893, upper bound: 0.0061887
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061966, upper bound: 0.0061813
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061889, upper bound: 0.0061884
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061952, upper bound: 0.0061824
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061883, upper bound: 0.0061906
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061974, upper bound: 0.0061799
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061899, upper bound: 0.0061869
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061960, upper bound: 0.0061808
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061892, upper bound: 0.0061887
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061900, upper bound: 0.0061889
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061816, upper bound: 0.0061952
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061879, upper bound: 0.0061897
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061805, upper bound: 0.0061968
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061911, upper bound: 0.0061874
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061830, upper bound: 0.0061941
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061888, upper bound: 0.0061882
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061800, upper bound: 0.0061953
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061900, upper bound: 0.0061889
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061814, upper bound: 0.0061951
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061879, upper bound: 0.0061897
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061803, upper bound: 0.0061967
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061911, upper bound: 0.0061874
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061826, upper bound: 0.0061939
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061888, upper bound: 0.0061882
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061800, upper bound: 0.0061952
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061907, upper bound: 0.0061873
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061823, upper bound: 0.0061927
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061895, upper bound: 0.0061887
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061815, upper bound: 0.0061951
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061800, upper bound: 0.0061861
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061835, upper bound: 0.0061918
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061906, upper bound: 0.0061875
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061827, upper bound: 0.0061938
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061907, upper bound: 0.0061873
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061800, upper bound: 0.0061927
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061895, upper bound: 0.0061887
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061813, upper bound: 0.0061951
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061919, upper bound: 0.0061861
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061833, upper bound: 0.0061917
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061906, upper bound: 0.0061875
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061824, upper bound: 0.0061938
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061800, upper bound: 0.0061824
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061800, upper bound: 0.0061906
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061800, upper bound: 0.0061833
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061800, upper bound: 0.0061919
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061951, upper bound: 0.0061813
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061887, upper bound: 0.0061895
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061927, upper bound: 0.0061821
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061800, upper bound: 0.0061907
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061800, upper bound: 0.0061827
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061800, upper bound: 0.0061906
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061800, upper bound: 0.0061835
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061800, upper bound: 0.0061919
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061951, upper bound: 0.0061815
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061887, upper bound: 0.0061895
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061927, upper bound: 0.0061823
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061800, upper bound: 0.0061907
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061952, upper bound: 0.0061816
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061800, upper bound: 0.0061888
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061939, upper bound: 0.0061826
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061873, upper bound: 0.0061910
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061968, upper bound: 0.0061804
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061897, upper bound: 0.0061879
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061951, upper bound: 0.0061814
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061889, upper bound: 0.0061900
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061953, upper bound: 0.0061819
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061882, upper bound: 0.0061888
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061941, upper bound: 0.0061830
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061874, upper bound: 0.0061910
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061968, upper bound: 0.0061806
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061897, upper bound: 0.0061879
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061952, upper bound: 0.0061816
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061888, upper bound: 0.0061900
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061800, upper bound: 0.0061893
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061800, upper bound: 0.0061959
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061800, upper bound: 0.0061899
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061799, upper bound: 0.0061974
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061906, upper bound: 0.0061883
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061824, upper bound: 0.0061953
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061884, upper bound: 0.0061890
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061800, upper bound: 0.0061966
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061800, upper bound: 0.0061893
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061800, upper bound: 0.0061960
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061870, upper bound: 0.0061900
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061799, upper bound: 0.0061974
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061907, upper bound: 0.0061884
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.95
Output dim: 7, lower bound: -0.0061824, upper bound: 0.0061953
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062150
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062111
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062139
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062107
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062132
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062111
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062140
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 7, lower bound: -0.0062047, upper bound: 0.0062107
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.95
Output dim: 7, lower bound: -0.0062076, upper bound: 0.0062132

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 4.40 + 598.46 = 602.87 seconds
