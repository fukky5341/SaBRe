## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_8.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 2)
Time budget: 420 seconds
Split limit: 100
Threshold: 1.5895e-05


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261)
1: (-0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480)
2: (-0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418)
3: (-0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863)
4: (-0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.01 + 0.56 = 1.56 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0000187, upper bound: 0.0000186

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000162, upper bound: 0.0000162
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000162, upper bound: 0.0000162
time: 0.14 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.38 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.38
Output dim: 0, lower bound: -0.0000162, upper bound: 0.0000162
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.38
Output dim: 0, lower bound: -0.0000162, upper bound: 0.0000162

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000160, upper bound: 0.0000162
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000162, upper bound: 0.0000158
time: 0.14 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000158, upper bound: 0.0000162
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000162, upper bound: 0.0000160
time: 0.14 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.28 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.28
Output dim: 0, lower bound: -0.0000160, upper bound: 0.0000162
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.28
Output dim: 0, lower bound: -0.0000162, upper bound: 0.0000158
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.28
Output dim: 0, lower bound: -0.0000158, upper bound: 0.0000162
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.28
Output dim: 0, lower bound: -0.0000162, upper bound: 0.0000160

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000158, upper bound: 0.0000162
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000160, upper bound: 0.0000161
time: 0.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000161, upper bound: 0.0000158
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000162, upper bound: 0.0000157
time: 0.14 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000158, upper bound: 0.0000161
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000158, upper bound: 0.0000161
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000161, upper bound: 0.0000159
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000162, upper bound: 0.0000157
time: 0.14 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 1.30 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.30
Output dim: 0, lower bound: -0.0000158, upper bound: 0.0000162
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.30
Output dim: 0, lower bound: -0.0000160, upper bound: 0.0000161
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.30
Output dim: 0, lower bound: -0.0000161, upper bound: 0.0000158
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.30
Output dim: 0, lower bound: -0.0000162, upper bound: 0.0000157
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.30
Output dim: 0, lower bound: -0.0000158, upper bound: 0.0000161
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.30
Output dim: 0, lower bound: -0.0000158, upper bound: 0.0000161
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.30
Output dim: 0, lower bound: -0.0000161, upper bound: 0.0000159
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.30
Output dim: 0, lower bound: -0.0000162, upper bound: 0.0000157

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000148, upper bound: 0.0000158
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000157, upper bound: 0.0000160
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000158, upper bound: 0.0000158
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000158, upper bound: 0.0000160
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000160, upper bound: 0.0000157
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000159, upper bound: 0.0000156
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000161, upper bound: 0.0000157
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000158, upper bound: 0.0000148
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000148, upper bound: 0.0000158
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000157, upper bound: 0.0000160
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000157, upper bound: 0.0000158
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000157, upper bound: 0.0000160
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000160, upper bound: 0.0000157
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000158, upper bound: 0.0000158
time: 0.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000161, upper bound: 0.0000157
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000158, upper bound: 0.0000148
time: 0.15 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 1.31 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 1.31
Output dim: 0, lower bound: -0.0000148, upper bound: 0.0000158
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.31
Output dim: 0, lower bound: -0.0000157, upper bound: 0.0000160
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 1.31
Output dim: 0, lower bound: -0.0000158, upper bound: 0.0000158
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.31
Output dim: 0, lower bound: -0.0000158, upper bound: 0.0000160
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.31
Output dim: 0, lower bound: -0.0000160, upper bound: 0.0000157
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 1.31
Output dim: 0, lower bound: -0.0000159, upper bound: 0.0000156
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.31
Output dim: 0, lower bound: -0.0000161, upper bound: 0.0000157
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 1.31
Output dim: 0, lower bound: -0.0000158, upper bound: 0.0000148
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 1.31
Output dim: 0, lower bound: -0.0000148, upper bound: 0.0000158
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.31
Output dim: 0, lower bound: -0.0000157, upper bound: 0.0000160
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 1.31
Output dim: 0, lower bound: -0.0000157, upper bound: 0.0000158
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.31
Output dim: 0, lower bound: -0.0000157, upper bound: 0.0000160
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.31
Output dim: 0, lower bound: -0.0000160, upper bound: 0.0000157
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 1.31
Output dim: 0, lower bound: -0.0000158, upper bound: 0.0000158
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.31
Output dim: 0, lower bound: -0.0000161, upper bound: 0.0000157
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 1.31
Output dim: 0, lower bound: -0.0000158, upper bound: 0.0000148

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000150, upper bound: 0.0000152
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000150, upper bound: 0.0000151
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000151, upper bound: 0.0000152
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000150, upper bound: 0.0000151
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000152, upper bound: 0.0000149
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000143, upper bound: 0.0000148
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000152, upper bound: 0.0000149
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000149, upper bound: 0.0000143
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000143, upper bound: 0.0000149
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000150, upper bound: 0.0000151
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000148, upper bound: 0.0000143
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000150, upper bound: 0.0000152
time: 0.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000152, upper bound: 0.0000150
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000152, upper bound: 0.0000151
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261
1: -0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480
2: -0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418
3: -0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863
4: -0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000152, upper bound: 0.0000150
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000152, upper bound: 0.0000150
time: 0.16 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 1.35 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.35
Output dim: 0, lower bound: -0.0000150, upper bound: 0.0000152
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.35
Output dim: 0, lower bound: -0.0000150, upper bound: 0.0000151
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.35
Output dim: 0, lower bound: -0.0000151, upper bound: 0.0000152
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.35
Output dim: 0, lower bound: -0.0000150, upper bound: 0.0000151
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.35
Output dim: 0, lower bound: -0.0000152, upper bound: 0.0000149
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.35
Output dim: 0, lower bound: -0.0000143, upper bound: 0.0000148
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.35
Output dim: 0, lower bound: -0.0000152, upper bound: 0.0000149
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.35
Output dim: 0, lower bound: -0.0000149, upper bound: 0.0000143
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.35
Output dim: 0, lower bound: -0.0000143, upper bound: 0.0000149
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.35
Output dim: 0, lower bound: -0.0000150, upper bound: 0.0000151
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.35
Output dim: 0, lower bound: -0.0000148, upper bound: 0.0000143
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.35
Output dim: 0, lower bound: -0.0000150, upper bound: 0.0000152
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.35
Output dim: 0, lower bound: -0.0000152, upper bound: 0.0000150
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.35
Output dim: 0, lower bound: -0.0000152, upper bound: 0.0000151
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.35
Output dim: 0, lower bound: -0.0000152, upper bound: 0.0000150
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.35
Output dim: 0, lower bound: -0.0000152, upper bound: 0.0000150

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 1.56 + 29.07 = 30.63 seconds
