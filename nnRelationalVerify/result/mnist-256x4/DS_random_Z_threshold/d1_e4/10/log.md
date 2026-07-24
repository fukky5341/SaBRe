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
execution time: IAR + RelationalAnalysis = 0.98 + 2.94 = 3.92 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0062749, upper bound: 0.0062749

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062610, upper bound: 0.0062610
time: 2.04 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062609, upper bound: 0.0062609
time: 1.97 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 4.02 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 4.02
Output dim: 7, lower bound: -0.0062610, upper bound: 0.0062610
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 4.02
Output dim: 7, lower bound: -0.0062609, upper bound: 0.0062609

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

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062491, upper bound: 0.0062491
time: 2.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062491, upper bound: 0.0062491
time: 1.92 seconds

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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062293, upper bound: 0.0062254
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062254, upper bound: 0.0062293
time: 1.72 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.86 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.86
Output dim: 7, lower bound: -0.0062491, upper bound: 0.0062491
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.86
Output dim: 7, lower bound: -0.0062491, upper bound: 0.0062491
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.86
Output dim: 7, lower bound: -0.0062293, upper bound: 0.0062254
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.86
Output dim: 7, lower bound: -0.0062254, upper bound: 0.0062293

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

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062345, upper bound: 0.0062345
time: 1.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062345, upper bound: 0.0062345
time: 1.99 seconds

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

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0059789, upper bound: 0.0059790
time: 1.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0059789, upper bound: 0.0059790
time: 1.37 seconds

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 186

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062116, upper bound: 0.0061737
time: 1.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061757, upper bound: 0.0062075
time: 1.91 seconds

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

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 127

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062170, upper bound: 0.0062208
time: 1.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062177, upper bound: 0.0062202
time: 1.36 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.75 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.75
Output dim: 7, lower bound: -0.0062345, upper bound: 0.0062345
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.75
Output dim: 7, lower bound: -0.0062345, upper bound: 0.0062345
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.75
Output dim: 7, lower bound: -0.0059789, upper bound: 0.0059790
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.75
Output dim: 7, lower bound: -0.0059789, upper bound: 0.0059790
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.75
Output dim: 7, lower bound: -0.0062116, upper bound: 0.0061737
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.75
Output dim: 7, lower bound: -0.0061757, upper bound: 0.0062075
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.75
Output dim: 7, lower bound: -0.0062170, upper bound: 0.0062208
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.75
Output dim: 7, lower bound: -0.0062177, upper bound: 0.0062202

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

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062309, upper bound: 0.0062334
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062334, upper bound: 0.0062309
time: 1.90 seconds

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

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062270, upper bound: 0.0062271
time: 1.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062271, upper bound: 0.0062270
time: 1.51 seconds

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

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 127

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0059498, upper bound: 0.0059510
time: 1.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0059510, upper bound: 0.0059498
time: 1.50 seconds

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

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0058977, upper bound: 0.0058886
time: 1.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0058886, upper bound: 0.0058977
time: 1.44 seconds

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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0060994, upper bound: 0.0060994
time: 1.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061349, upper bound: 0.0060994
time: 1.35 seconds

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

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061632, upper bound: 0.0061971
time: 1.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061637, upper bound: 0.0061968
time: 1.75 seconds

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

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0055916, upper bound: 0.0055919
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0055916, upper bound: 0.0055919
time: 1.03 seconds

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

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062060, upper bound: 0.0062073
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062064, upper bound: 0.0062072
time: 1.71 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 4.22 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 7, lower bound: -0.0062309, upper bound: 0.0062334
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 7, lower bound: -0.0062334, upper bound: 0.0062309
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 7, lower bound: -0.0062270, upper bound: 0.0062271
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 7, lower bound: -0.0062271, upper bound: 0.0062270
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 4.22
Output dim: 7, lower bound: -0.0059498, upper bound: 0.0059510
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 4.22
Output dim: 7, lower bound: -0.0059510, upper bound: 0.0059498
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 4.22
Output dim: 7, lower bound: -0.0058977, upper bound: 0.0058886
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 4.22
Output dim: 7, lower bound: -0.0058886, upper bound: 0.0058977
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 7, lower bound: -0.0060994, upper bound: 0.0060994
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 7, lower bound: -0.0061349, upper bound: 0.0060994
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 7, lower bound: -0.0061632, upper bound: 0.0061971
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 7, lower bound: -0.0061637, upper bound: 0.0061968
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 4.22
Output dim: 7, lower bound: -0.0055916, upper bound: 0.0055919
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 4.22
Output dim: 7, lower bound: -0.0055916, upper bound: 0.0055919
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 7, lower bound: -0.0062060, upper bound: 0.0062073
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.22
Output dim: 7, lower bound: -0.0062064, upper bound: 0.0062072

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

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0060906, upper bound: 0.0060916
time: 1.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0060906, upper bound: 0.0060916
time: 1.67 seconds

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

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 127

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062262, upper bound: 0.0062243
time: 1.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062268, upper bound: 0.0062238
time: 2.12 seconds

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

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062175, upper bound: 0.0062175
time: 2.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062175, upper bound: 0.0062192
time: 2.21 seconds

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

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 127

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062197, upper bound: 0.0062203
time: 1.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062197, upper bound: 0.0062197
time: 1.73 seconds

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

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 127

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061058, upper bound: 0.0060705
time: 1.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061061, upper bound: 0.0060685
time: 2.09 seconds

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

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 185

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061314, upper bound: 0.0060825
time: 1.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061143, upper bound: 0.0060959
time: 1.40 seconds

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

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 185

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061599, upper bound: 0.0061781
time: 1.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061455, upper bound: 0.0061938
time: 1.36 seconds

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

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 127

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061522, upper bound: 0.0061878
time: 2.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061541, upper bound: 0.0061875
time: 1.50 seconds

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

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0056018, upper bound: 0.0056018
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0056018, upper bound: 0.0056018
time: 1.20 seconds

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

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061982, upper bound: 0.0061984
time: 1.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061975, upper bound: 0.0061991
time: 1.96 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 4.74 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 7, lower bound: -0.0060906, upper bound: 0.0060916
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 7, lower bound: -0.0060906, upper bound: 0.0060916
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 7, lower bound: -0.0062262, upper bound: 0.0062243
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 7, lower bound: -0.0062268, upper bound: 0.0062238
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 7, lower bound: -0.0062175, upper bound: 0.0062175
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 7, lower bound: -0.0062175, upper bound: 0.0062192
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 7, lower bound: -0.0062197, upper bound: 0.0062203
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 7, lower bound: -0.0062197, upper bound: 0.0062197
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 7, lower bound: -0.0061058, upper bound: 0.0060705
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 7, lower bound: -0.0061061, upper bound: 0.0060685
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 7, lower bound: -0.0061314, upper bound: 0.0060825
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 7, lower bound: -0.0061143, upper bound: 0.0060959
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 7, lower bound: -0.0061599, upper bound: 0.0061781
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 7, lower bound: -0.0061455, upper bound: 0.0061938
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 7, lower bound: -0.0061522, upper bound: 0.0061878
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 7, lower bound: -0.0061541, upper bound: 0.0061875
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.74
Output dim: 7, lower bound: -0.0056018, upper bound: 0.0056018
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.74
Output dim: 7, lower bound: -0.0056018, upper bound: 0.0056018
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 7, lower bound: -0.0061982, upper bound: 0.0061984
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 7, lower bound: -0.0061975, upper bound: 0.0061991

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

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 185

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0060868, upper bound: 0.0060690
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0060678, upper bound: 0.0060879
time: 1.71 seconds

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

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0042754, upper bound: 0.0042756
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0042754, upper bound: 0.0042756
time: 0.82 seconds

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

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062089, upper bound: 0.0062010
time: 1.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062004, upper bound: 0.0062073
time: 1.61 seconds

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

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0060153, upper bound: 0.0060145
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0060153, upper bound: 0.0060145
time: 1.64 seconds

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

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0059896, upper bound: 0.0059896
time: 1.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0059896, upper bound: 0.0059896
time: 2.00 seconds

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

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 186

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061635, upper bound: 0.0061675
time: 2.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061635, upper bound: 0.0062021
time: 2.03 seconds

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

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 186

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 185

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062030, upper bound: 0.0062038
time: 1.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0062030, upper bound: 0.0062172
time: 1.68 seconds

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

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 186

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061652, upper bound: 0.0061657
time: 2.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061652, upper bound: 0.0062030
time: 1.64 seconds

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

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0054535, upper bound: 0.0054320
time: 1.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0054535, upper bound: 0.0054320
time: 1.47 seconds

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

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0060060, upper bound: 0.0060060
time: 1.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0060426, upper bound: 0.0060068
time: 1.38 seconds

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

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0044455, upper bound: 0.0044266
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0044455, upper bound: 0.0044266
time: 0.85 seconds

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

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0053020, upper bound: 0.0052866
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0053020, upper bound: 0.0052866
time: 1.15 seconds

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

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0055195, upper bound: 0.0055384
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0055195, upper bound: 0.0055384
time: 1.45 seconds

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

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061192, upper bound: 0.0061684
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061188, upper bound: 0.0061697
time: 1.96 seconds

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

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0054927, upper bound: 0.0055221
time: 1.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0054939, upper bound: 0.0055221
time: 1.48 seconds

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

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061468, upper bound: 0.0061854
time: 1.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061520, upper bound: 0.0061845
time: 1.51 seconds

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

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061681, upper bound: 0.0061610
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061620, upper bound: 0.0061667
time: 1.42 seconds

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

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061292, upper bound: 0.0061296
time: 1.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061292, upper bound: 0.0061296
time: 1.47 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 4.09 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 7, lower bound: -0.0060868, upper bound: 0.0060690
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 7, lower bound: -0.0060678, upper bound: 0.0060879
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.09
Output dim: 7, lower bound: -0.0042754, upper bound: 0.0042756
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.09
Output dim: 7, lower bound: -0.0042754, upper bound: 0.0042756
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 7, lower bound: -0.0062089, upper bound: 0.0062010
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 7, lower bound: -0.0062004, upper bound: 0.0062073
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 7, lower bound: -0.0060153, upper bound: 0.0060145
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 7, lower bound: -0.0060153, upper bound: 0.0060145
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 7, lower bound: -0.0059896, upper bound: 0.0059896
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 7, lower bound: -0.0059896, upper bound: 0.0059896
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 7, lower bound: -0.0061635, upper bound: 0.0061675
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 7, lower bound: -0.0061635, upper bound: 0.0062021
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 7, lower bound: -0.0062030, upper bound: 0.0062038
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 7, lower bound: -0.0062030, upper bound: 0.0062172
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 7, lower bound: -0.0061652, upper bound: 0.0061657
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 7, lower bound: -0.0061652, upper bound: 0.0062030
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.09
Output dim: 7, lower bound: -0.0054535, upper bound: 0.0054320
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.09
Output dim: 7, lower bound: -0.0054535, upper bound: 0.0054320
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 7, lower bound: -0.0060060, upper bound: 0.0060060
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 7, lower bound: -0.0060426, upper bound: 0.0060068
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.09
Output dim: 7, lower bound: -0.0044455, upper bound: 0.0044266
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.09
Output dim: 7, lower bound: -0.0044455, upper bound: 0.0044266
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.09
Output dim: 7, lower bound: -0.0053020, upper bound: 0.0052866
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.09
Output dim: 7, lower bound: -0.0053020, upper bound: 0.0052866
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.09
Output dim: 7, lower bound: -0.0055195, upper bound: 0.0055384
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.09
Output dim: 7, lower bound: -0.0055195, upper bound: 0.0055384
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 7, lower bound: -0.0061192, upper bound: 0.0061684
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 7, lower bound: -0.0061188, upper bound: 0.0061697
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.09
Output dim: 7, lower bound: -0.0054927, upper bound: 0.0055221
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.09
Output dim: 7, lower bound: -0.0054939, upper bound: 0.0055221
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 7, lower bound: -0.0061468, upper bound: 0.0061854
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 7, lower bound: -0.0061520, upper bound: 0.0061845
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 7, lower bound: -0.0061681, upper bound: 0.0061610
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 7, lower bound: -0.0061620, upper bound: 0.0061667
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 7, lower bound: -0.0061292, upper bound: 0.0061296
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.09
Output dim: 7, lower bound: -0.0061292, upper bound: 0.0061296

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

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0042719, upper bound: 0.0042683
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0042719, upper bound: 0.0042683
time: 0.84 seconds

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

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 186

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0060473, upper bound: 0.0060314
time: 1.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0060158, upper bound: 0.0060665
time: 1.62 seconds

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

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 186

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061949, upper bound: 0.0061868
time: 2.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061859, upper bound: 0.0061871
time: 2.09 seconds

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

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0051923, upper bound: 0.0051973
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0051923, upper bound: 0.0051973
time: 1.23 seconds

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

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0051948, upper bound: 0.0051945
time: 1.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0051948, upper bound: 0.0051945
time: 1.42 seconds

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

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0060075, upper bound: 0.0060042
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0060046, upper bound: 0.0060068
time: 1.36 seconds

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

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0042150, upper bound: 0.0042099
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0042150, upper bound: 0.0042099
time: 0.82 seconds

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

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 186

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0059719, upper bound: 0.0059335
time: 1.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0059327, upper bound: 0.0059691
time: 1.75 seconds

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

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0059770, upper bound: 0.0059789
time: 1.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0059770, upper bound: 0.0059789
time: 1.68 seconds

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

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0059327, upper bound: 0.0059721
time: 1.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0059327, upper bound: 0.0059721
time: 1.58 seconds

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

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061943, upper bound: 0.0061951
time: 2.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061943, upper bound: 0.0061953
time: 2.15 seconds

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

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 186

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061993, upper bound: 0.0062161
time: 1.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061993, upper bound: 0.0062135
time: 2.20 seconds

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

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0051983, upper bound: 0.0051779
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0051983, upper bound: 0.0051779
time: 1.09 seconds

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

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061516, upper bound: 0.0061873
time: 2.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061516, upper bound: 0.0061892
time: 2.14 seconds

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

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0043641, upper bound: 0.0043530
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0043641, upper bound: 0.0043530
time: 0.89 seconds

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

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0043641, upper bound: 0.0043532
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0043641, upper bound: 0.0043532
time: 0.85 seconds

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

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0054958, upper bound: 0.0055345
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0054958, upper bound: 0.0055345
time: 1.34 seconds

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

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0054343, upper bound: 0.0054746
time: 1.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0054343, upper bound: 0.0054746
time: 1.41 seconds

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

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0054881, upper bound: 0.0055155
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0054881, upper bound: 0.0055155
time: 1.17 seconds

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

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0054887, upper bound: 0.0055155
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0054887, upper bound: 0.0055155
time: 1.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 186

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0060990, upper bound: 0.0060992
time: 2.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061081, upper bound: 0.0061008
time: 1.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 186

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061420, upper bound: 0.0061106
time: 1.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061096, upper bound: 0.0061476
time: 2.02 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061213, upper bound: 0.0061232
time: 1.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061227, upper bound: 0.0061226
time: 1.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 185

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061091, upper bound: 0.0061094
time: 1.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061095, upper bound: 0.0061261
time: 1.42 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 3.93 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.93
Output dim: 7, lower bound: -0.0042719, upper bound: 0.0042683
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.93
Output dim: 7, lower bound: -0.0042719, upper bound: 0.0042683
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 7, lower bound: -0.0060473, upper bound: 0.0060314
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 7, lower bound: -0.0060158, upper bound: 0.0060665
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 7, lower bound: -0.0061949, upper bound: 0.0061868
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 7, lower bound: -0.0061859, upper bound: 0.0061871
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.93
Output dim: 7, lower bound: -0.0051923, upper bound: 0.0051973
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.93
Output dim: 7, lower bound: -0.0051923, upper bound: 0.0051973
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.93
Output dim: 7, lower bound: -0.0051948, upper bound: 0.0051945
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.93
Output dim: 7, lower bound: -0.0051948, upper bound: 0.0051945
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 7, lower bound: -0.0060075, upper bound: 0.0060042
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 7, lower bound: -0.0060046, upper bound: 0.0060068
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.93
Output dim: 7, lower bound: -0.0042150, upper bound: 0.0042099
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.93
Output dim: 7, lower bound: -0.0042150, upper bound: 0.0042099
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 7, lower bound: -0.0059719, upper bound: 0.0059335
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 7, lower bound: -0.0059327, upper bound: 0.0059691
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 7, lower bound: -0.0059770, upper bound: 0.0059789
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 7, lower bound: -0.0059770, upper bound: 0.0059789
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 7, lower bound: -0.0059327, upper bound: 0.0059721
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 7, lower bound: -0.0059327, upper bound: 0.0059721
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 7, lower bound: -0.0061943, upper bound: 0.0061951
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 7, lower bound: -0.0061943, upper bound: 0.0061953
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 7, lower bound: -0.0061993, upper bound: 0.0062161
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 7, lower bound: -0.0061993, upper bound: 0.0062135
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.93
Output dim: 7, lower bound: -0.0051983, upper bound: 0.0051779
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.93
Output dim: 7, lower bound: -0.0051983, upper bound: 0.0051779
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 7, lower bound: -0.0061516, upper bound: 0.0061873
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 7, lower bound: -0.0061516, upper bound: 0.0061892
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.93
Output dim: 7, lower bound: -0.0043641, upper bound: 0.0043530
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.93
Output dim: 7, lower bound: -0.0043641, upper bound: 0.0043530
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.93
Output dim: 7, lower bound: -0.0043641, upper bound: 0.0043532
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.93
Output dim: 7, lower bound: -0.0043641, upper bound: 0.0043532
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.93
Output dim: 7, lower bound: -0.0054958, upper bound: 0.0055345
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.93
Output dim: 7, lower bound: -0.0054958, upper bound: 0.0055345
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.93
Output dim: 7, lower bound: -0.0054343, upper bound: 0.0054746
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.93
Output dim: 7, lower bound: -0.0054343, upper bound: 0.0054746
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.93
Output dim: 7, lower bound: -0.0054881, upper bound: 0.0055155
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.93
Output dim: 7, lower bound: -0.0054881, upper bound: 0.0055155
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.93
Output dim: 7, lower bound: -0.0054887, upper bound: 0.0055155
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.93
Output dim: 7, lower bound: -0.0054887, upper bound: 0.0055155
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 7, lower bound: -0.0060990, upper bound: 0.0060992
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 7, lower bound: -0.0061081, upper bound: 0.0061008
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 7, lower bound: -0.0061420, upper bound: 0.0061106
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 7, lower bound: -0.0061096, upper bound: 0.0061476
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 7, lower bound: -0.0061213, upper bound: 0.0061232
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 7, lower bound: -0.0061227, upper bound: 0.0061226
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 7, lower bound: -0.0061091, upper bound: 0.0061094
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.93
Output dim: 7, lower bound: -0.0061095, upper bound: 0.0061261

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0060061, upper bound: 0.0060218
time: 1.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0060061, upper bound: 0.0060234
time: 1.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0042344, upper bound: 0.0042434
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0042344, upper bound: 0.0042434
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 186

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061354, upper bound: 0.0061377
time: 2.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061412, upper bound: 0.0061684
time: 2.03 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0051343, upper bound: 0.0051305
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0051343, upper bound: 0.0051305
time: 1.01 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0042296, upper bound: 0.0042247
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0042296, upper bound: 0.0042247
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0059202, upper bound: 0.0059212
time: 1.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0059206, upper bound: 0.0059293
time: 1.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0058949, upper bound: 0.0058546
time: 1.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0058865, upper bound: 0.0058575
time: 1.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0041786, upper bound: 0.0041818
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0041786, upper bound: 0.0041818
time: 0.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 127

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0059826, upper bound: 0.0059500
time: 1.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0059463, upper bound: 0.0059477
time: 1.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0059681, upper bound: 0.0059734
time: 2.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0059681, upper bound: 0.0059706
time: 1.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0058566, upper bound: 0.0058868
time: 1.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0058531, upper bound: 0.0058952
time: 1.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0059257, upper bound: 0.0059656
time: 1.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0059262, upper bound: 0.0059655
time: 1.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 186

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061911, upper bound: 0.0061784
time: 1.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061780, upper bound: 0.0061794
time: 1.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0058176, upper bound: 0.0058052
time: 1.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0058176, upper bound: 0.0058052
time: 1.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0059451, upper bound: 0.0059612
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0059451, upper bound: 0.0059612
time: 1.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061852, upper bound: 0.0061995
time: 2.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061852, upper bound: 0.0061998
time: 2.13 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061365, upper bound: 0.0061620
time: 1.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061302, upper bound: 0.0061696
time: 2.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0051155, upper bound: 0.0051362
time: 1.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0051155, upper bound: 0.0051362
time: 1.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0054467, upper bound: 0.0054366
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0054467, upper bound: 0.0054366
time: 1.33 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061004, upper bound: 0.0060943
time: 1.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0060916, upper bound: 0.0060932
time: 1.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0054304, upper bound: 0.0054057
time: 1.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0054304, upper bound: 0.0054057
time: 1.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0049094, upper bound: 0.0049331
time: 1.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0049094, upper bound: 0.0049331
time: 1.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0060389, upper bound: 0.0060401
time: 1.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0060399, upper bound: 0.0060471
time: 1.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 185

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061022, upper bound: 0.0061025
time: 1.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061030, upper bound: 0.0061191
time: 1.95 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0053867, upper bound: 0.0053765
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0053867, upper bound: 0.0053765
time: 1.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 186

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0060890, upper bound: 0.0060667
time: 1.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0060534, upper bound: 0.0061061
time: 1.88 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 4.59 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0060061, upper bound: 0.0060218
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0060061, upper bound: 0.0060234
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0042344, upper bound: 0.0042434
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0042344, upper bound: 0.0042434
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0061354, upper bound: 0.0061377
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0061412, upper bound: 0.0061684
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0051343, upper bound: 0.0051305
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0051343, upper bound: 0.0051305
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0042296, upper bound: 0.0042247
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0042296, upper bound: 0.0042247
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0059202, upper bound: 0.0059212
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0059206, upper bound: 0.0059293
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0058949, upper bound: 0.0058546
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0058865, upper bound: 0.0058575
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0041786, upper bound: 0.0041818
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0041786, upper bound: 0.0041818
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0059826, upper bound: 0.0059500
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0059463, upper bound: 0.0059477
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0059681, upper bound: 0.0059734
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0059681, upper bound: 0.0059706
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0058566, upper bound: 0.0058868
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0058531, upper bound: 0.0058952
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0059257, upper bound: 0.0059656
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0059262, upper bound: 0.0059655
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0061911, upper bound: 0.0061784
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0061780, upper bound: 0.0061794
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0058176, upper bound: 0.0058052
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0058176, upper bound: 0.0058052
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0059451, upper bound: 0.0059612
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0059451, upper bound: 0.0059612
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0061852, upper bound: 0.0061995
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0061852, upper bound: 0.0061998
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0061365, upper bound: 0.0061620
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0061302, upper bound: 0.0061696
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0051155, upper bound: 0.0051362
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0051155, upper bound: 0.0051362
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0054467, upper bound: 0.0054366
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0054467, upper bound: 0.0054366
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0061004, upper bound: 0.0060943
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0060916, upper bound: 0.0060932
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0054304, upper bound: 0.0054057
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0054304, upper bound: 0.0054057
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0049094, upper bound: 0.0049331
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0049094, upper bound: 0.0049331
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0060389, upper bound: 0.0060401
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0060399, upper bound: 0.0060471
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0061022, upper bound: 0.0061025
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0061030, upper bound: 0.0061191
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0053867, upper bound: 0.0053765
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0053867, upper bound: 0.0053765
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0060890, upper bound: 0.0060667
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.59
Output dim: 7, lower bound: -0.0060534, upper bound: 0.0061061

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0042305, upper bound: 0.0042284
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0042305, upper bound: 0.0042284
time: 1.14 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0042290, upper bound: 0.0042300
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0042290, upper bound: 0.0042300
time: 0.88 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0058947, upper bound: 0.0058520
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0058487, upper bound: 0.0058520
time: 1.97 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0056902, upper bound: 0.0057106
time: 1.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0056902, upper bound: 0.0057106
time: 1.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0051186, upper bound: 0.0050980
time: 1.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0051186, upper bound: 0.0050980
time: 1.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 185

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0060009, upper bound: 0.0059547
time: 1.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0059845, upper bound: 0.0059696
time: 1.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0041864, upper bound: 0.0041840
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0041864, upper bound: 0.0041840
time: 0.88 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 185

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0059222, upper bound: 0.0059452
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0059115, upper bound: 0.0059621
time: 1.34 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 127

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0058956, upper bound: 0.0059364
time: 1.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0058973, upper bound: 0.0059361
time: 1.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 186

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061270, upper bound: 0.0061285
time: 1.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061269, upper bound: 0.0061603
time: 1.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 186

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0057212, upper bound: 0.0057231
time: 1.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0057212, upper bound: 0.0057231
time: 1.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0057497, upper bound: 0.0057661
time: 1.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0057497, upper bound: 0.0057661
time: 1.94 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0058820, upper bound: 0.0058993
time: 1.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0058822, upper bound: 0.0058992
time: 1.94 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0057765, upper bound: 0.0058000
time: 1.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0057765, upper bound: 0.0058000
time: 1.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061209, upper bound: 0.0061603
time: 2.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061209, upper bound: 0.0061615
time: 2.13 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0054387, upper bound: 0.0054308
time: 2.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0054387, upper bound: 0.0054308
time: 1.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0048794, upper bound: 0.0048765
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0048794, upper bound: 0.0048765
time: 1.10 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0058771, upper bound: 0.0058711
time: 1.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0058771, upper bound: 0.0058711
time: 1.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0059707, upper bound: 0.0059792
time: 1.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0059724, upper bound: 0.0059805
time: 1.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 186

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0053806, upper bound: 0.0053700
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0053806, upper bound: 0.0053700
time: 1.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 186

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0060442, upper bound: 0.0060593
time: 1.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0060470, upper bound: 0.0060992
time: 1.70 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0053583, upper bound: 0.0053491
time: 1.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0053583, upper bound: 0.0053491
time: 1.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0060442, upper bound: 0.0060997
time: 1.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0060470, upper bound: 0.0060992
time: 1.69 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 4.67 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 7, lower bound: -0.0042305, upper bound: 0.0042284
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 7, lower bound: -0.0042305, upper bound: 0.0042284
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 7, lower bound: -0.0042290, upper bound: 0.0042300
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 7, lower bound: -0.0042290, upper bound: 0.0042300
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 7, lower bound: -0.0058947, upper bound: 0.0058520
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 7, lower bound: -0.0058487, upper bound: 0.0058520
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 7, lower bound: -0.0056902, upper bound: 0.0057106
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 7, lower bound: -0.0056902, upper bound: 0.0057106
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 7, lower bound: -0.0051186, upper bound: 0.0050980
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 7, lower bound: -0.0051186, upper bound: 0.0050980
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 4.67
Output dim: 7, lower bound: -0.0060009, upper bound: 0.0059547
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 4.67
Output dim: 7, lower bound: -0.0059845, upper bound: 0.0059696
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 7, lower bound: -0.0041864, upper bound: 0.0041840
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 7, lower bound: -0.0041864, upper bound: 0.0041840
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 7, lower bound: -0.0059222, upper bound: 0.0059452
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 4.67
Output dim: 7, lower bound: -0.0059115, upper bound: 0.0059621
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 7, lower bound: -0.0058956, upper bound: 0.0059364
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 7, lower bound: -0.0058973, upper bound: 0.0059361
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 4.67
Output dim: 7, lower bound: -0.0061270, upper bound: 0.0061285
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 4.67
Output dim: 7, lower bound: -0.0061269, upper bound: 0.0061603
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 7, lower bound: -0.0057212, upper bound: 0.0057231
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 7, lower bound: -0.0057212, upper bound: 0.0057231
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 7, lower bound: -0.0057497, upper bound: 0.0057661
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 7, lower bound: -0.0057497, upper bound: 0.0057661
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 7, lower bound: -0.0058820, upper bound: 0.0058993
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 7, lower bound: -0.0058822, upper bound: 0.0058992
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 7, lower bound: -0.0057765, upper bound: 0.0058000
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 7, lower bound: -0.0057765, upper bound: 0.0058000
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 4.67
Output dim: 7, lower bound: -0.0061209, upper bound: 0.0061603
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 4.67
Output dim: 7, lower bound: -0.0061209, upper bound: 0.0061615
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 7, lower bound: -0.0054387, upper bound: 0.0054308
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 7, lower bound: -0.0054387, upper bound: 0.0054308
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 7, lower bound: -0.0048794, upper bound: 0.0048765
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 7, lower bound: -0.0048794, upper bound: 0.0048765
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 7, lower bound: -0.0058771, upper bound: 0.0058711
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 7, lower bound: -0.0058771, upper bound: 0.0058711
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 4.67
Output dim: 7, lower bound: -0.0059707, upper bound: 0.0059792
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 4.67
Output dim: 7, lower bound: -0.0059724, upper bound: 0.0059805
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 7, lower bound: -0.0053806, upper bound: 0.0053700
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 7, lower bound: -0.0053806, upper bound: 0.0053700
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 4.67
Output dim: 7, lower bound: -0.0060442, upper bound: 0.0060593
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 4.67
Output dim: 7, lower bound: -0.0060470, upper bound: 0.0060992
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 7, lower bound: -0.0053583, upper bound: 0.0053491
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.67
Output dim: 7, lower bound: -0.0053583, upper bound: 0.0053491
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 4.67
Output dim: 7, lower bound: -0.0060442, upper bound: 0.0060997
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 4.67
Output dim: 7, lower bound: -0.0060470, upper bound: 0.0060992

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0059394, upper bound: 0.0058933
time: 1.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0058909, upper bound: 0.0058933
time: 1.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0058690, upper bound: 0.0058785
time: 1.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0059005, upper bound: 0.0058854
time: 1.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 127

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0058816, upper bound: 0.0059330
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0058825, upper bound: 0.0059327
time: 1.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061126, upper bound: 0.0061144
time: 2.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061126, upper bound: 0.0061139
time: 1.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061230, upper bound: 0.0061591
time: 2.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0061359, upper bound: 0.0061570
time: 1.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0056249, upper bound: 0.0056592
time: 1.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0056249, upper bound: 0.0056592
time: 1.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0056249, upper bound: 0.0056613
time: 1.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0056249, upper bound: 0.0056613
time: 1.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 185

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0059695, upper bound: 0.0059608
time: 1.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0059597, upper bound: 0.0059757
time: 1.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 185

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0059597, upper bound: 0.0059621
time: 1.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0059597, upper bound: 0.0059770
time: 1.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0053517, upper bound: 0.0053417
time: 1.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0053517, upper bound: 0.0053417
time: 1.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0047806, upper bound: 0.0048120
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0047806, upper bound: 0.0048120
time: 1.03 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0058816, upper bound: 0.0059325
time: 1.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0058816, upper bound: 0.0059325
time: 1.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0053294, upper bound: 0.0053665
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0053294, upper bound: 0.0053665
time: 1.06 seconds

## Summary of splitting (split count: 9)
- Time for DS candidates: 3.27 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 3.27
Output dim: 7, lower bound: -0.0059394, upper bound: 0.0058933
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 3.27
Output dim: 7, lower bound: -0.0058909, upper bound: 0.0058933
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 3.27
Output dim: 7, lower bound: -0.0058690, upper bound: 0.0058785
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 3.27
Output dim: 7, lower bound: -0.0059005, upper bound: 0.0058854
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 3.27
Output dim: 7, lower bound: -0.0058816, upper bound: 0.0059330
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 3.27
Output dim: 7, lower bound: -0.0058825, upper bound: 0.0059327
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 3.27
Output dim: 7, lower bound: -0.0061126, upper bound: 0.0061144
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 3.27
Output dim: 7, lower bound: -0.0061126, upper bound: 0.0061139
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 3.27
Output dim: 7, lower bound: -0.0061230, upper bound: 0.0061591
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 3.27
Output dim: 7, lower bound: -0.0061359, upper bound: 0.0061570
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 3.27
Output dim: 7, lower bound: -0.0056249, upper bound: 0.0056592
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 3.27
Output dim: 7, lower bound: -0.0056249, upper bound: 0.0056592
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 3.27
Output dim: 7, lower bound: -0.0056249, upper bound: 0.0056613
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 3.27
Output dim: 7, lower bound: -0.0056249, upper bound: 0.0056613
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 3.27
Output dim: 7, lower bound: -0.0059695, upper bound: 0.0059608
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 3.27
Output dim: 7, lower bound: -0.0059597, upper bound: 0.0059757
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 3.27
Output dim: 7, lower bound: -0.0059597, upper bound: 0.0059621
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 3.27
Output dim: 7, lower bound: -0.0059597, upper bound: 0.0059770
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 3.27
Output dim: 7, lower bound: -0.0053517, upper bound: 0.0053417
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 3.27
Output dim: 7, lower bound: -0.0053517, upper bound: 0.0053417
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 3.27
Output dim: 7, lower bound: -0.0047806, upper bound: 0.0048120
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 3.27
Output dim: 7, lower bound: -0.0047806, upper bound: 0.0048120
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 3.27
Output dim: 7, lower bound: -0.0058816, upper bound: 0.0059325
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 3.27
Output dim: 7, lower bound: -0.0058816, upper bound: 0.0059325
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 3.27
Output dim: 7, lower bound: -0.0053294, upper bound: 0.0053665
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 3.27
Output dim: 7, lower bound: -0.0053294, upper bound: 0.0053665

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0058345, upper bound: 0.0057862
time: 1.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0058345, upper bound: 0.0057862
time: 1.36 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0057508, upper bound: 0.0057524
time: 2.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0057508, upper bound: 0.0057524
time: 1.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0050901, upper bound: 0.0050955
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0050901, upper bound: 0.0050955
time: 1.10 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0050903, upper bound: 0.0050955
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0050903, upper bound: 0.0050955
time: 1.11 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 186

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0059497, upper bound: 0.0059046
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0059035, upper bound: 0.0059404
time: 1.89 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0052334, upper bound: 0.0052472
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0052334, upper bound: 0.0052472
time: 1.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 186

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0059487, upper bound: 0.0059049
time: 1.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0059035, upper bound: 0.0059418
time: 1.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0046913, upper bound: 0.0047007
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0046913, upper bound: 0.0047007
time: 1.33 seconds

## Summary of splitting (split count: 10)
- Time for DS candidates: 3.85 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 11, time: 3.85
Output dim: 7, lower bound: -0.0058345, upper bound: 0.0057862
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 11, time: 3.85
Output dim: 7, lower bound: -0.0058345, upper bound: 0.0057862
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 11, time: 3.85
Output dim: 7, lower bound: -0.0057508, upper bound: 0.0057524
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 11, time: 3.85
Output dim: 7, lower bound: -0.0057508, upper bound: 0.0057524
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 11, time: 3.85
Output dim: 7, lower bound: -0.0050901, upper bound: 0.0050955
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 11, time: 3.85
Output dim: 7, lower bound: -0.0050901, upper bound: 0.0050955
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 11, time: 3.85
Output dim: 7, lower bound: -0.0050903, upper bound: 0.0050955
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 11, time: 3.85
Output dim: 7, lower bound: -0.0050903, upper bound: 0.0050955
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 11, time: 3.85
Output dim: 7, lower bound: -0.0059497, upper bound: 0.0059046
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 11, time: 3.85
Output dim: 7, lower bound: -0.0059035, upper bound: 0.0059404
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 11, time: 3.85
Output dim: 7, lower bound: -0.0052334, upper bound: 0.0052472
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 11, time: 3.85
Output dim: 7, lower bound: -0.0052334, upper bound: 0.0052472
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 11, time: 3.85
Output dim: 7, lower bound: -0.0059487, upper bound: 0.0059049
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 11, time: 3.85
Output dim: 7, lower bound: -0.0059035, upper bound: 0.0059418
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 11, time: 3.85
Output dim: 7, lower bound: -0.0046913, upper bound: 0.0047007
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 11, time: 3.85
Output dim: 7, lower bound: -0.0046913, upper bound: 0.0047007

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.92 + 577.49 = 581.40 seconds
