## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_7.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 6)
Time budget: 420 seconds
Split limit: 100
Threshold: 1129.173098456097


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475)
1: (-326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244)
2: (-470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781)
3: (-276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557)
4: (-433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.86 + 2.55 = 3.41 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -1129.1843903, upper bound: 1129.1843903

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1840692, upper bound: 1129.1840692
time: 1.42 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1840692, upper bound: 1129.1840692
time: 1.37 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.87 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 2.87
Output dim: 3, lower bound: -1129.1840692, upper bound: 1129.1840692
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 2.87
Output dim: 3, lower bound: -1129.1840692, upper bound: 1129.1840692

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836275, upper bound: 1129.1836275
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836275, upper bound: 1129.1836275
time: 1.06 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836275, upper bound: 1129.1836275
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836275, upper bound: 1129.1836275
time: 1.06 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.99 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.99
Output dim: 3, lower bound: -1129.1836275, upper bound: 1129.1836275
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.99
Output dim: 3, lower bound: -1129.1836275, upper bound: 1129.1836275
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.99
Output dim: 3, lower bound: -1129.1836275, upper bound: 1129.1836275
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.99
Output dim: 3, lower bound: -1129.1836275, upper bound: 1129.1836275

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1787841, upper bound: 1129.1787841
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1787841, upper bound: 1129.1787841
time: 0.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1787841, upper bound: 1129.1787841
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1787841, upper bound: 1129.1787841
time: 1.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1787841, upper bound: 1129.1787841
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1787841, upper bound: 1129.1787841
time: 0.87 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1787841, upper bound: 1129.1787841
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1787841, upper bound: 1129.1787841
time: 0.87 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.65 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 3, lower bound: -1129.1787841, upper bound: 1129.1787841
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 3, lower bound: -1129.1787841, upper bound: 1129.1787841
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 3, lower bound: -1129.1787841, upper bound: 1129.1787841
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 3, lower bound: -1129.1787841, upper bound: 1129.1787841
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 3, lower bound: -1129.1787841, upper bound: 1129.1787841
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 3, lower bound: -1129.1787841, upper bound: 1129.1787841
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 3, lower bound: -1129.1787841, upper bound: 1129.1787841
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 3, lower bound: -1129.1787841, upper bound: 1129.1787841

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1787428, upper bound: 1129.1787428
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1787428, upper bound: 1129.1787841
time: 0.94 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1787428, upper bound: 1129.1787428
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1787428, upper bound: 1129.1787841
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1787428, upper bound: 1129.1787428
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1787428, upper bound: 1129.1787841
time: 0.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1787841, upper bound: 1129.1787428
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1787428, upper bound: 1129.1787841
time: 1.06 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1787428, upper bound: 1129.1787428
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1787428, upper bound: 1129.1787841
time: 0.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1787428, upper bound: 1129.1787428
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1787428, upper bound: 1129.1787841
time: 0.91 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1787841, upper bound: 1129.1787428
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1787428, upper bound: 1129.1787841
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1787428, upper bound: 1129.1787428
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1787428, upper bound: 1129.1787841
time: 0.94 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 3.29 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 3, lower bound: -1129.1787428, upper bound: 1129.1787428
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 3, lower bound: -1129.1787428, upper bound: 1129.1787841
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 3, lower bound: -1129.1787428, upper bound: 1129.1787428
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 3, lower bound: -1129.1787428, upper bound: 1129.1787841
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 3, lower bound: -1129.1787428, upper bound: 1129.1787428
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 3, lower bound: -1129.1787428, upper bound: 1129.1787841
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 3, lower bound: -1129.1787841, upper bound: 1129.1787428
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 3, lower bound: -1129.1787428, upper bound: 1129.1787841
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 3, lower bound: -1129.1787428, upper bound: 1129.1787428
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 3, lower bound: -1129.1787428, upper bound: 1129.1787841
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 3, lower bound: -1129.1787428, upper bound: 1129.1787428
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 3, lower bound: -1129.1787428, upper bound: 1129.1787841
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 3, lower bound: -1129.1787841, upper bound: 1129.1787428
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 3, lower bound: -1129.1787428, upper bound: 1129.1787841
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 3, lower bound: -1129.1787428, upper bound: 1129.1787428
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 3, lower bound: -1129.1787428, upper bound: 1129.1787841

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786226, upper bound: 1129.1786074
time: 1.02 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786226
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786851
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786851, upper bound: 1129.1786074
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786226, upper bound: 1129.1786074
time: 0.91 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786226
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786851
time: 1.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786851, upper bound: 1129.1786074
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786226, upper bound: 1129.1786074
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786226
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786851
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786851, upper bound: 1129.1786074
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786226, upper bound: 1129.1786074
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786226
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786851
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786226, upper bound: 1129.1786074
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.87 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786226
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786851
time: 0.86 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786851, upper bound: 1129.1786074
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786226, upper bound: 1129.1786074
time: 1.10 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786226
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786851
time: 1.08 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786851, upper bound: 1129.1786074
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786226, upper bound: 1129.1786074
time: 0.91 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786226
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786851
time: 1.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786226, upper bound: 1129.1786074
time: 1.09 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786226
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786851
time: 0.91 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 3.27 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -1129.1786226, upper bound: 1129.1786074
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786226
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786851
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -1129.1786851, upper bound: 1129.1786074
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -1129.1786226, upper bound: 1129.1786074
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786226
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786851
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -1129.1786851, upper bound: 1129.1786074
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -1129.1786226, upper bound: 1129.1786074
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786226
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786851
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -1129.1786851, upper bound: 1129.1786074
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -1129.1786226, upper bound: 1129.1786074
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786226
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786851
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -1129.1786226, upper bound: 1129.1786074
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786226
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786851
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -1129.1786851, upper bound: 1129.1786074
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -1129.1786226, upper bound: 1129.1786074
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786226
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786851
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -1129.1786851, upper bound: 1129.1786074
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -1129.1786226, upper bound: 1129.1786074
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786226
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786851
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -1129.1786226, upper bound: 1129.1786074
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786226
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786851

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786851, upper bound: 1129.1786074
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786226, upper bound: 1129.1786074
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786226
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786097
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786851
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.94 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786851, upper bound: 1129.1786074
time: 0.94 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786097, upper bound: 1129.1786074
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786226, upper bound: 1129.1786074
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786226
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786097
time: 0.97 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786851
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.97 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.12 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786226
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786097
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786851
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.03 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786097, upper bound: 1129.1786074
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786226
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786097
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786851
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.08 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786226, upper bound: 1129.1786074
time: 1.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786097, upper bound: 1129.1786074
time: 1.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.05 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786226
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786097
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786851
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786851, upper bound: 1129.1786074
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786097, upper bound: 1129.1786074
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786226
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786097
time: 0.91 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786851
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.91 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786851, upper bound: 1129.1786074
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786097, upper bound: 1129.1786074
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786097, upper bound: 1129.1786074
time: 0.97 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786226
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786097
time: 1.02 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786851
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.11 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786097, upper bound: 1129.1786074
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786226, upper bound: 1129.1786074
time: 0.94 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786226
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786097
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786851
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.08 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 3.47 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786851, upper bound: 1129.1786074
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786226, upper bound: 1129.1786074
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786226
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786097
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786851
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786851, upper bound: 1129.1786074
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786097, upper bound: 1129.1786074
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786226, upper bound: 1129.1786074
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786226
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786097
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786851
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786226
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786097
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786851
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786097, upper bound: 1129.1786074
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786226
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786097
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786851
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786226, upper bound: 1129.1786074
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786097, upper bound: 1129.1786074
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786226
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786097
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786851
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786851, upper bound: 1129.1786074
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786097, upper bound: 1129.1786074
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786226
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786097
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786851
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786851, upper bound: 1129.1786074
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786097, upper bound: 1129.1786074
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786097, upper bound: 1129.1786074
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786226
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786097
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786851
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786097, upper bound: 1129.1786074
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786226, upper bound: 1129.1786074
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786226
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786097
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786851
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786097, upper bound: 1129.1786074
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786226, upper bound: 1129.1786074
time: 0.90 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786226
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.98 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786097
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.88 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786851
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.90 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.90 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786851, upper bound: 1129.1786074
time: 0.96 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786097, upper bound: 1129.1786074
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786226
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786097
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786851
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.08 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.00 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786097, upper bound: 1129.1786074
time: 1.01 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786097, upper bound: 1129.1786074
time: 1.35 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786226
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786097
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786851
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786851, upper bound: 1129.1786074
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786097, upper bound: 1129.1786074
time: 0.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786226, upper bound: 1129.1786074
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786226
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.98 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786097
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.01 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786851
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.05 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.91 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786097, upper bound: 1129.1786074
time: 0.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786226
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786097
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.95 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786851
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.95 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.06 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.97 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786097, upper bound: 1129.1786074
time: 0.92 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.50 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786226
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.92 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786097
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786851
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.07 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.93 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.91 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786097, upper bound: 1129.1786074
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.86 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786226
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.88 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786097
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786851
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.92 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786851, upper bound: 1129.1786074
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786097, upper bound: 1129.1786074
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786226, upper bound: 1129.1786074
time: 0.93 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786226
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.11 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786097
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.03 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786851
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.09 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475
1: -326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244
2: -470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781
3: -276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557
4: -433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
time: 1.06 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 3.44 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786097, upper bound: 1129.1786074
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786226, upper bound: 1129.1786074
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786226
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786097
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786851
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786851, upper bound: 1129.1786074
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786097, upper bound: 1129.1786074
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786226
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786097
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786851
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786097, upper bound: 1129.1786074
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786097, upper bound: 1129.1786074
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786226
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786097
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786851
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786851, upper bound: 1129.1786074
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786097, upper bound: 1129.1786074
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786226, upper bound: 1129.1786074
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786226
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786097
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786851
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786097, upper bound: 1129.1786074
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786226
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786097
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786851
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786097, upper bound: 1129.1786074
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786226
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786097
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786851
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786097, upper bound: 1129.1786074
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786226
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786097
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786851
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786851, upper bound: 1129.1786074
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786097, upper bound: 1129.1786074
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786226, upper bound: 1129.1786074
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786226
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786097
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786851
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 3, lower bound: -1129.1786074, upper bound: 1129.1786074

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 3.41 + 417.80 = 421.21 seconds
