## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_2.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 2)
Time budget: 420 seconds
Split limit: 100
Threshold: 2.978080836


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314)
1: (-1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077)
2: (-1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894)
3: (-2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013)
4: (-1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.40 + 0.93 = 2.33 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -2.9900410, upper bound: 2.9900410

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9877534, upper bound: 2.9877534
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9877534, upper bound: 2.9877534
time: 0.25 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.63 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.63
Output dim: 0, lower bound: -2.9877534, upper bound: 2.9877534
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.63
Output dim: 0, lower bound: -2.9877534, upper bound: 2.9877534

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9856746, upper bound: 2.9856746
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9856746, upper bound: 2.9856746
time: 0.24 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9856746, upper bound: 2.9856746
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9856746, upper bound: 2.9856746
time: 0.24 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.92 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.92
Output dim: 0, lower bound: -2.9856746, upper bound: 2.9856746
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.92
Output dim: 0, lower bound: -2.9856746, upper bound: 2.9856746
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.92
Output dim: 0, lower bound: -2.9856746, upper bound: 2.9856746
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.92
Output dim: 0, lower bound: -2.9856746, upper bound: 2.9856746

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9831259, upper bound: 2.9830196
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9830196, upper bound: 2.9831427
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9831427, upper bound: 2.9830196
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9830196, upper bound: 2.9831259
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9830196, upper bound: 2.9830196
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9830196, upper bound: 2.9831427
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9831427, upper bound: 2.9830196
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9830196, upper bound: 2.9831259
time: 0.26 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 1.97 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.97
Output dim: 0, lower bound: -2.9831259, upper bound: 2.9830196
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.97
Output dim: 0, lower bound: -2.9830196, upper bound: 2.9831427
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.97
Output dim: 0, lower bound: -2.9831427, upper bound: 2.9830196
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.97
Output dim: 0, lower bound: -2.9830196, upper bound: 2.9831259
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.97
Output dim: 0, lower bound: -2.9830196, upper bound: 2.9830196
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.97
Output dim: 0, lower bound: -2.9830196, upper bound: 2.9831427
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.97
Output dim: 0, lower bound: -2.9831427, upper bound: 2.9830196
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.97
Output dim: 0, lower bound: -2.9830196, upper bound: 2.9831259

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9821054, upper bound: 2.9820158
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9820158, upper bound: 2.9820212
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9820212, upper bound: 2.9821773
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9820158, upper bound: 2.9821224
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9821224, upper bound: 2.9820158
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9821773, upper bound: 2.9820212
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9820212, upper bound: 2.9821188
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9820158, upper bound: 2.9821054
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9821054, upper bound: 2.9820158
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9821188, upper bound: 2.9820212
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9820212, upper bound: 2.9821773
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9820158, upper bound: 2.9821224
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9821224, upper bound: 2.9820158
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9821773, upper bound: 2.9820212
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9820212, upper bound: 2.9821188
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9820158, upper bound: 2.9821054
time: 0.26 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.11 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.11
Output dim: 0, lower bound: -2.9821054, upper bound: 2.9820158
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.11
Output dim: 0, lower bound: -2.9820158, upper bound: 2.9820212
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.11
Output dim: 0, lower bound: -2.9820212, upper bound: 2.9821773
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.11
Output dim: 0, lower bound: -2.9820158, upper bound: 2.9821224
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.11
Output dim: 0, lower bound: -2.9821224, upper bound: 2.9820158
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.11
Output dim: 0, lower bound: -2.9821773, upper bound: 2.9820212
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.11
Output dim: 0, lower bound: -2.9820212, upper bound: 2.9821188
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.11
Output dim: 0, lower bound: -2.9820158, upper bound: 2.9821054
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.11
Output dim: 0, lower bound: -2.9821054, upper bound: 2.9820158
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.11
Output dim: 0, lower bound: -2.9821188, upper bound: 2.9820212
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.11
Output dim: 0, lower bound: -2.9820212, upper bound: 2.9821773
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.11
Output dim: 0, lower bound: -2.9820158, upper bound: 2.9821224
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.11
Output dim: 0, lower bound: -2.9821224, upper bound: 2.9820158
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.11
Output dim: 0, lower bound: -2.9821773, upper bound: 2.9820212
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.11
Output dim: 0, lower bound: -2.9820212, upper bound: 2.9821188
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.11
Output dim: 0, lower bound: -2.9820158, upper bound: 2.9821054

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811837, upper bound: 2.9811287
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811837, upper bound: 2.9810397
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811836, upper bound: 2.9811313
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811730, upper bound: 2.9810122
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811525, upper bound: 2.9813299
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811341, upper bound: 2.9811865
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811255, upper bound: 2.9812511
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811290, upper bound: 2.9811842
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811842, upper bound: 2.9811290
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9812511, upper bound: 2.9811258
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811865, upper bound: 2.9811341
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9813299, upper bound: 2.9811525
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9810949, upper bound: 2.9811830
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811313, upper bound: 2.9811836
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9810397, upper bound: 2.9811837
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811287, upper bound: 2.9811837
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811837, upper bound: 2.9811287
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811837, upper bound: 2.9810882
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811836, upper bound: 2.9811313
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811290, upper bound: 2.9810949
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811525, upper bound: 2.9813299
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811341, upper bound: 2.9811865
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811258, upper bound: 2.9812511
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811290, upper bound: 2.9811842
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811842, upper bound: 2.9811290
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811313, upper bound: 2.9811255
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811865, upper bound: 2.9811341
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811287, upper bound: 2.9811525
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9810122, upper bound: 2.9811730
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811313, upper bound: 2.9811836
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9810397, upper bound: 2.9811837
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811287, upper bound: 2.9811837
time: 0.26 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 2.99 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -2.9811837, upper bound: 2.9811287
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -2.9811837, upper bound: 2.9810397
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -2.9811836, upper bound: 2.9811313
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -2.9811730, upper bound: 2.9810122
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -2.9811525, upper bound: 2.9813299
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -2.9811341, upper bound: 2.9811865
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -2.9811255, upper bound: 2.9812511
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -2.9811290, upper bound: 2.9811842
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -2.9811842, upper bound: 2.9811290
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -2.9812511, upper bound: 2.9811258
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -2.9811865, upper bound: 2.9811341
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -2.9813299, upper bound: 2.9811525
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -2.9810949, upper bound: 2.9811830
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -2.9811313, upper bound: 2.9811836
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -2.9810397, upper bound: 2.9811837
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -2.9811287, upper bound: 2.9811837
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -2.9811837, upper bound: 2.9811287
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -2.9811837, upper bound: 2.9810882
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -2.9811836, upper bound: 2.9811313
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -2.9811290, upper bound: 2.9810949
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -2.9811525, upper bound: 2.9813299
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -2.9811341, upper bound: 2.9811865
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -2.9811258, upper bound: 2.9812511
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -2.9811290, upper bound: 2.9811842
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -2.9811842, upper bound: 2.9811290
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -2.9811313, upper bound: 2.9811255
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -2.9811865, upper bound: 2.9811341
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -2.9811287, upper bound: 2.9811525
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -2.9810122, upper bound: 2.9811730
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -2.9811313, upper bound: 2.9811836
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -2.9810397, upper bound: 2.9811837
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -2.9811287, upper bound: 2.9811837

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811837, upper bound: 2.9811287
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811565, upper bound: 2.9811276
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811837, upper bound: 2.9810397
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811399, upper bound: 2.9810122
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811836, upper bound: 2.9811298
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811834, upper bound: 2.9811313
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811730, upper bound: 2.9810122
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811651, upper bound: 2.9810122
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811525, upper bound: 2.9813299
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811288, upper bound: 2.9813092
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811341, upper bound: 2.9811865
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811306, upper bound: 2.9811849
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9810122, upper bound: 2.9811383
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811255, upper bound: 2.9812511
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9810122, upper bound: 2.9810122
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811290, upper bound: 2.9811842
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811842, upper bound: 2.9811290
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811259, upper bound: 2.9811283
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9812511, upper bound: 2.9811257
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811383, upper bound: 2.9811258
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811849, upper bound: 2.9811306
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811865, upper bound: 2.9811341
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9813092, upper bound: 2.9811288
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9813299, upper bound: 2.9811525
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9810949, upper bound: 2.9811821
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9810923, upper bound: 2.9811830
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811313, upper bound: 2.9811834
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811298, upper bound: 2.9811836
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9810122, upper bound: 2.9811399
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9810882, upper bound: 2.9811837
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811269, upper bound: 2.9811565
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811287, upper bound: 2.9811837
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811837, upper bound: 2.9811287
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811565, upper bound: 2.9811269
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811837, upper bound: 2.9810882
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811399, upper bound: 2.9810122
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811836, upper bound: 2.9811298
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811834, upper bound: 2.9811313
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811830, upper bound: 2.9810923
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811821, upper bound: 2.9810949
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811525, upper bound: 2.9813299
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811288, upper bound: 2.9813092
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811341, upper bound: 2.9811865
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811306, upper bound: 2.9811849
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811258, upper bound: 2.9811383
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811257, upper bound: 2.9812511
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811283, upper bound: 2.9811259
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811290, upper bound: 2.9811842
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811842, upper bound: 2.9811290
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9810122, upper bound: 2.9810122
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9812511, upper bound: 2.9811255
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811298, upper bound: 2.9810122
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811849, upper bound: 2.9811306
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811865, upper bound: 2.9811341
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9813092, upper bound: 2.9811288
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9813299, upper bound: 2.9811525
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9810122, upper bound: 2.9811651
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9810122, upper bound: 2.9811730
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811313, upper bound: 2.9811834
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811298, upper bound: 2.9811836
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9810122, upper bound: 2.9811399
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9810397, upper bound: 2.9811837
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811276, upper bound: 2.9811565
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811287, upper bound: 2.9811837
time: 0.26 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.52 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9811837, upper bound: 2.9811287
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9811565, upper bound: 2.9811276
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9811837, upper bound: 2.9810397
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9811399, upper bound: 2.9810122
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9811836, upper bound: 2.9811298
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9811834, upper bound: 2.9811313
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9811730, upper bound: 2.9810122
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9811651, upper bound: 2.9810122
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9811525, upper bound: 2.9813299
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9811288, upper bound: 2.9813092
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9811341, upper bound: 2.9811865
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9811306, upper bound: 2.9811849
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9810122, upper bound: 2.9811383
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9811255, upper bound: 2.9812511
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9810122, upper bound: 2.9810122
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9811290, upper bound: 2.9811842
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9811842, upper bound: 2.9811290
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9811259, upper bound: 2.9811283
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9812511, upper bound: 2.9811257
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9811383, upper bound: 2.9811258
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9811849, upper bound: 2.9811306
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9811865, upper bound: 2.9811341
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9813092, upper bound: 2.9811288
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9813299, upper bound: 2.9811525
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9810949, upper bound: 2.9811821
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9810923, upper bound: 2.9811830
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9811313, upper bound: 2.9811834
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9811298, upper bound: 2.9811836
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9810122, upper bound: 2.9811399
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9810882, upper bound: 2.9811837
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9811269, upper bound: 2.9811565
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9811287, upper bound: 2.9811837
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9811837, upper bound: 2.9811287
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9811565, upper bound: 2.9811269
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9811837, upper bound: 2.9810882
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9811399, upper bound: 2.9810122
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9811836, upper bound: 2.9811298
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9811834, upper bound: 2.9811313
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9811830, upper bound: 2.9810923
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9811821, upper bound: 2.9810949
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9811525, upper bound: 2.9813299
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9811288, upper bound: 2.9813092
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9811341, upper bound: 2.9811865
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9811306, upper bound: 2.9811849
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9811258, upper bound: 2.9811383
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9811257, upper bound: 2.9812511
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9811283, upper bound: 2.9811259
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9811290, upper bound: 2.9811842
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9811842, upper bound: 2.9811290
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9810122, upper bound: 2.9810122
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9812511, upper bound: 2.9811255
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9811298, upper bound: 2.9810122
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9811849, upper bound: 2.9811306
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9811865, upper bound: 2.9811341
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9813092, upper bound: 2.9811288
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9813299, upper bound: 2.9811525
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9810122, upper bound: 2.9811651
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9810122, upper bound: 2.9811730
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9811313, upper bound: 2.9811834
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9811298, upper bound: 2.9811836
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9810122, upper bound: 2.9811399
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9810397, upper bound: 2.9811837
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9811276, upper bound: 2.9811565
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.9811287, upper bound: 2.9811837

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.55 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.56 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.55 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.56 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.56 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.56 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.56 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.56 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.56 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675341
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.57 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.57 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.57 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.57 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.57 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.57 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.57 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.57 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.58 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.57 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.57 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.58 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.58 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.58 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.58 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675341, upper bound: 2.9674900
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.52 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.58 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.51 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.58 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.52 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.58 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.52 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.58 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.53 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.59 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.52 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.59 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.60 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.53 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.59 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.53 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.59 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.59 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.59 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.60 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.60 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.60 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.60 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.57 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.60 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.60 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675341
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.60 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.61 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.60 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.60 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.60 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.61 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.61 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.62 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.61 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.61 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.61 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.61 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.61 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.62 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.62 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675341, upper bound: 2.9674900
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.62 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.63 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.62 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.62 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.62 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.63 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.63 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.62 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.30 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 3.28 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675341
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675341, upper bound: 2.9674900
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675341
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675341, upper bound: 2.9674900
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 2.33 + 351.35 = 353.69 seconds
