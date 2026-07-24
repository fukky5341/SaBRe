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
execution time: IAR + RelationalAnalysis = 0.56 + 0.85 = 1.41 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -2.9900410, upper bound: 2.9900410

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9892933, upper bound: 2.9892933
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9892933, upper bound: 2.9892933
time: 0.18 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.39 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.39
Output dim: 0, lower bound: -2.9892933, upper bound: 2.9892933
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.39
Output dim: 0, lower bound: -2.9892933, upper bound: 2.9892933

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9869833, upper bound: 2.9869833
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9869833, upper bound: 2.9869833
time: 0.17 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9842031, upper bound: 2.9842031
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9842031, upper bound: 2.9842031
time: 0.17 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 0.88 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 0.88
Output dim: 0, lower bound: -2.9869833, upper bound: 2.9869833
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 0.88
Output dim: 0, lower bound: -2.9869833, upper bound: 2.9869833
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 0.88
Output dim: 0, lower bound: -2.9842031, upper bound: 2.9842031
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 0.88
Output dim: 0, lower bound: -2.9842031, upper bound: 2.9842031

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9831259, upper bound: 2.9830196
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9831427, upper bound: 2.9830196
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9831259, upper bound: 2.9830196
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9831427, upper bound: 2.9830196
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9830196, upper bound: 2.9831427
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9830196, upper bound: 2.9831427
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9830196, upper bound: 2.9831259
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9830196, upper bound: 2.9831259
time: 0.19 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 0.94 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 0.94
Output dim: 0, lower bound: -2.9831259, upper bound: 2.9830196
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 0.94
Output dim: 0, lower bound: -2.9831427, upper bound: 2.9830196
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 0.94
Output dim: 0, lower bound: -2.9831259, upper bound: 2.9830196
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 0.94
Output dim: 0, lower bound: -2.9831427, upper bound: 2.9830196
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 0.94
Output dim: 0, lower bound: -2.9830196, upper bound: 2.9831427
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 0.94
Output dim: 0, lower bound: -2.9830196, upper bound: 2.9831427
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 0.94
Output dim: 0, lower bound: -2.9830196, upper bound: 2.9831259
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 0.94
Output dim: 0, lower bound: -2.9830196, upper bound: 2.9831259

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.53 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9821582, upper bound: 2.9821602
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9821806, upper bound: 2.9820036
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.54 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9822013, upper bound: 2.9821616
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9821602, upper bound: 2.9821582
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.54 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9821970, upper bound: 2.9821602
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9821921, upper bound: 2.9821017
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.55 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9831075, upper bound: 2.9830174
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9830174, upper bound: 2.9830196
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.53 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9821582, upper bound: 2.9823346
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9821616, upper bound: 2.9822013
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.54 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9830196, upper bound: 2.9831427
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9830174, upper bound: 2.9831075
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.55 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9820212, upper bound: 2.9821188
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9820158, upper bound: 2.9821054
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.55 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9830196, upper bound: 2.9831259
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9830174, upper bound: 2.9831056
time: 0.21 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 1.15 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.15
Output dim: 0, lower bound: -2.9821582, upper bound: 2.9821602
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.15
Output dim: 0, lower bound: -2.9821806, upper bound: 2.9820036
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.15
Output dim: 0, lower bound: -2.9822013, upper bound: 2.9821616
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.15
Output dim: 0, lower bound: -2.9821602, upper bound: 2.9821582
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.15
Output dim: 0, lower bound: -2.9821970, upper bound: 2.9821602
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.15
Output dim: 0, lower bound: -2.9821921, upper bound: 2.9821017
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.15
Output dim: 0, lower bound: -2.9831075, upper bound: 2.9830174
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.15
Output dim: 0, lower bound: -2.9830174, upper bound: 2.9830196
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.15
Output dim: 0, lower bound: -2.9821582, upper bound: 2.9823346
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.15
Output dim: 0, lower bound: -2.9821616, upper bound: 2.9822013
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.15
Output dim: 0, lower bound: -2.9830196, upper bound: 2.9831427
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.15
Output dim: 0, lower bound: -2.9830174, upper bound: 2.9831075
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.15
Output dim: 0, lower bound: -2.9820212, upper bound: 2.9821188
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.15
Output dim: 0, lower bound: -2.9820158, upper bound: 2.9821054
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.15
Output dim: 0, lower bound: -2.9830196, upper bound: 2.9831259
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.15
Output dim: 0, lower bound: -2.9830174, upper bound: 2.9831056

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.55 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811837, upper bound: 2.9811287
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811836, upper bound: 2.9811313
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.54 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811837, upper bound: 2.9810397
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811730, upper bound: 2.9810122
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.55 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811842, upper bound: 2.9811290
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811865, upper bound: 2.9811341
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.54 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9822762, upper bound: 2.9821543
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9821580, upper bound: 2.9821582
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.54 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

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
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811837, upper bound: 2.9811287
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811836, upper bound: 2.9811313
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.55 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811837, upper bound: 2.9810882
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811290, upper bound: 2.9810949
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.56 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9821224, upper bound: 2.9820158
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9821678, upper bound: 2.9820170
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.55 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9820036, upper bound: 2.9821616
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9821580, upper bound: 2.9821582
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.56 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9821582, upper bound: 2.9823346
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9821543, upper bound: 2.9822762
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.56 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9821602, upper bound: 2.9822013
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9821588, upper bound: 2.9821726
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9821582, upper bound: 2.9823346
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9821616, upper bound: 2.9822013
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.56 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9820170, upper bound: 2.9821678
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9820158, upper bound: 2.9821224
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.56 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9820212, upper bound: 2.9821188
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9820170, upper bound: 2.9821133
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.56 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9810397, upper bound: 2.9811837
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811287, upper bound: 2.9811837
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9820212, upper bound: 2.9821188
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9820152, upper bound: 2.9820475
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9820170, upper bound: 2.9821133
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9820158, upper bound: 2.9821054
time: 0.19 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 1.19 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.19
Output dim: 0, lower bound: -2.9811837, upper bound: 2.9811287
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.19
Output dim: 0, lower bound: -2.9811836, upper bound: 2.9811313
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.19
Output dim: 0, lower bound: -2.9811837, upper bound: 2.9810397
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.19
Output dim: 0, lower bound: -2.9811730, upper bound: 2.9810122
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.19
Output dim: 0, lower bound: -2.9811842, upper bound: 2.9811290
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.19
Output dim: 0, lower bound: -2.9811865, upper bound: 2.9811341
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.19
Output dim: 0, lower bound: -2.9822762, upper bound: 2.9821543
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.19
Output dim: 0, lower bound: -2.9821580, upper bound: 2.9821582
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.19
Output dim: 0, lower bound: -2.9811837, upper bound: 2.9811287
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.19
Output dim: 0, lower bound: -2.9811836, upper bound: 2.9811313
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.19
Output dim: 0, lower bound: -2.9811837, upper bound: 2.9810882
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.19
Output dim: 0, lower bound: -2.9811290, upper bound: 2.9810949
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.19
Output dim: 0, lower bound: -2.9821224, upper bound: 2.9820158
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.19
Output dim: 0, lower bound: -2.9821678, upper bound: 2.9820170
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.19
Output dim: 0, lower bound: -2.9820036, upper bound: 2.9821616
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.19
Output dim: 0, lower bound: -2.9821580, upper bound: 2.9821582
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.19
Output dim: 0, lower bound: -2.9821582, upper bound: 2.9823346
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.19
Output dim: 0, lower bound: -2.9821543, upper bound: 2.9822762
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.19
Output dim: 0, lower bound: -2.9821602, upper bound: 2.9822013
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.19
Output dim: 0, lower bound: -2.9821588, upper bound: 2.9821726
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.19
Output dim: 0, lower bound: -2.9821582, upper bound: 2.9823346
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.19
Output dim: 0, lower bound: -2.9821616, upper bound: 2.9822013
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.19
Output dim: 0, lower bound: -2.9820170, upper bound: 2.9821678
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.19
Output dim: 0, lower bound: -2.9820158, upper bound: 2.9821224
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.19
Output dim: 0, lower bound: -2.9820212, upper bound: 2.9821188
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.19
Output dim: 0, lower bound: -2.9820170, upper bound: 2.9821133
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.19
Output dim: 0, lower bound: -2.9810397, upper bound: 2.9811837
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.19
Output dim: 0, lower bound: -2.9811287, upper bound: 2.9811837
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.19
Output dim: 0, lower bound: -2.9820212, upper bound: 2.9821188
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.19
Output dim: 0, lower bound: -2.9820152, upper bound: 2.9820475
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.19
Output dim: 0, lower bound: -2.9820170, upper bound: 2.9821133
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.19
Output dim: 0, lower bound: -2.9820158, upper bound: 2.9821054

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.56 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811837, upper bound: 2.9811287
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811565, upper bound: 2.9811276
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811836, upper bound: 2.9811298
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811834, upper bound: 2.9811313
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811837, upper bound: 2.9810397
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811399, upper bound: 2.9810122
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.56 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811730, upper bound: 2.9810122
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811651, upper bound: 2.9810122
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811842, upper bound: 2.9811290
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811259, upper bound: 2.9811283
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811849, upper bound: 2.9811306
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811865, upper bound: 2.9811341
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9812511, upper bound: 2.9811257
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9813092, upper bound: 2.9811288
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.58 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811383, upper bound: 2.9811258
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9813299, upper bound: 2.9811525
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811837, upper bound: 2.9811287
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811565, upper bound: 2.9811269
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.58 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811836, upper bound: 2.9811298
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811834, upper bound: 2.9811313
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811837, upper bound: 2.9810882
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811399, upper bound: 2.9810122
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811830, upper bound: 2.9810923
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811821, upper bound: 2.9810949
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811842, upper bound: 2.9811290
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9812511, upper bound: 2.9811255
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811849, upper bound: 2.9811306
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9813092, upper bound: 2.9811288
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9810122, upper bound: 2.9810122
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811865, upper bound: 2.9811341
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811383, upper bound: 2.9810122
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9813299, upper bound: 2.9811525
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811525, upper bound: 2.9813299
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9810122, upper bound: 2.9811383
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811288, upper bound: 2.9813092
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811255, upper bound: 2.9812511
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811341, upper bound: 2.9811865
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9810122, upper bound: 2.9810122
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811306, upper bound: 2.9811849
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811290, upper bound: 2.9811842
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811525, upper bound: 2.9813299
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811258, upper bound: 2.9811383
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811341, upper bound: 2.9811865
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811283, upper bound: 2.9811259
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811288, upper bound: 2.9813092
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811306, upper bound: 2.9811849
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811257, upper bound: 2.9812511
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811290, upper bound: 2.9811842
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9810949, upper bound: 2.9811821
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811313, upper bound: 2.9811834
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9810923, upper bound: 2.9811830
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811298, upper bound: 2.9811836
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9810122, upper bound: 2.9811399
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9810882, upper bound: 2.9811837
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811269, upper bound: 2.9811565
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811287, upper bound: 2.9811837
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9810122, upper bound: 2.9811651
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811313, upper bound: 2.9811834
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9810122, upper bound: 2.9811399
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811276, upper bound: 2.9811565
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9810122, upper bound: 2.9811730
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811298, upper bound: 2.9811836
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9810397, upper bound: 2.9811837
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811287, upper bound: 2.9811837
time: 0.19 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 1.07 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9811837, upper bound: 2.9811287
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9811565, upper bound: 2.9811276
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9811836, upper bound: 2.9811298
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9811834, upper bound: 2.9811313
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9811837, upper bound: 2.9810397
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9811399, upper bound: 2.9810122
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9811730, upper bound: 2.9810122
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9811651, upper bound: 2.9810122
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9811842, upper bound: 2.9811290
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9811259, upper bound: 2.9811283
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9811849, upper bound: 2.9811306
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9811865, upper bound: 2.9811341
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9812511, upper bound: 2.9811257
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9813092, upper bound: 2.9811288
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9811383, upper bound: 2.9811258
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9813299, upper bound: 2.9811525
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9811837, upper bound: 2.9811287
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9811565, upper bound: 2.9811269
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9811836, upper bound: 2.9811298
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9811834, upper bound: 2.9811313
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9811837, upper bound: 2.9810882
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9811399, upper bound: 2.9810122
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9811830, upper bound: 2.9810923
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9811821, upper bound: 2.9810949
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9811842, upper bound: 2.9811290
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9812511, upper bound: 2.9811255
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9811849, upper bound: 2.9811306
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9813092, upper bound: 2.9811288
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9810122, upper bound: 2.9810122
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9811865, upper bound: 2.9811341
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9811383, upper bound: 2.9810122
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9813299, upper bound: 2.9811525
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9811525, upper bound: 2.9813299
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9810122, upper bound: 2.9811383
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9811288, upper bound: 2.9813092
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9811255, upper bound: 2.9812511
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9811341, upper bound: 2.9811865
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9810122, upper bound: 2.9810122
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9811306, upper bound: 2.9811849
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9811290, upper bound: 2.9811842
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9811525, upper bound: 2.9813299
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9811258, upper bound: 2.9811383
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9811341, upper bound: 2.9811865
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9811283, upper bound: 2.9811259
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9811288, upper bound: 2.9813092
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9811306, upper bound: 2.9811849
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9811257, upper bound: 2.9812511
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9811290, upper bound: 2.9811842
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9810949, upper bound: 2.9811821
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9811313, upper bound: 2.9811834
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9810923, upper bound: 2.9811830
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9811298, upper bound: 2.9811836
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9810122, upper bound: 2.9811399
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9810882, upper bound: 2.9811837
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9811269, upper bound: 2.9811565
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9811287, upper bound: 2.9811837
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9810122, upper bound: 2.9811651
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9811313, upper bound: 2.9811834
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9810122, upper bound: 2.9811399
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9811276, upper bound: 2.9811565
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9810122, upper bound: 2.9811730
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9811298, upper bound: 2.9811836
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9810397, upper bound: 2.9811837
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.07
Output dim: 0, lower bound: -2.9811287, upper bound: 2.9811837

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 41

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9802552, upper bound: 2.9805236
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9805284, upper bound: 2.9802399
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 43

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9801981, upper bound: 2.9805258
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9805413, upper bound: 2.9802471
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 2

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 32

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.42 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9810573, upper bound: 2.9810397
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811837, upper bound: 2.9810359
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 45

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.42 seconds

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9789822, upper bound: 2.9785239
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9789822, upper bound: 2.9785239
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 38

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 38

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9789636, upper bound: 2.9785239
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9789636, upper bound: 2.9785239
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 43

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 41

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 40

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9810927, upper bound: 2.9811306
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811849, upper bound: 2.9810575
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 45

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9810359, upper bound: 2.9811341
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811865, upper bound: 2.9810573
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.61 seconds

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9790046, upper bound: 2.9788312
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9790046, upper bound: 2.9788312
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 16

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.61 seconds

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9802389, upper bound: 2.9805239
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9807060, upper bound: 2.9802560
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 32

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 13

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9789511, upper bound: 2.9786698
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9789511, upper bound: 2.9786698
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 32

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675341, upper bound: 2.9674900
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 37

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 32

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9394535, upper bound: 2.9395878
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9395878, upper bound: 2.9394535
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 2

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9810559, upper bound: 2.9811313
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811834, upper bound: 2.9810559
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 32

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.76 seconds

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9789498, upper bound: 2.9789217
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9789498, upper bound: 2.9789217
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.76 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 43

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 1.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 1

Time for candidate selection: 1.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9394535, upper bound: 2.9395878
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9395878, upper bound: 2.9394535
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 45

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9810559, upper bound: 2.9811255
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9812511, upper bound: 2.9810559
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.39 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9810927, upper bound: 2.9811306
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811849, upper bound: 2.9810575
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39

Time for candidate selection: 0.39 seconds

### Candidate
type: DSZ, layer: 5, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9789303, upper bound: 2.9789692
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9789303, upper bound: 2.9789692
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 2

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9394535, upper bound: 2.9395878
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9395878, upper bound: 2.9394535
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 37

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9786537, upper bound: 2.9790017
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9786537, upper bound: 2.9790017
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 40

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9394535, upper bound: 2.9395878
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9395878, upper bound: 2.9394535
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 16

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9802560, upper bound: 2.9805493
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9807267, upper bound: 2.9802561
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 41

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675341
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 37

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 5, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 16

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9789692, upper bound: 2.9789303
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9789692, upper bound: 2.9789303
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 40

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9810559, upper bound: 2.9812511
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811255, upper bound: 2.9810807
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 38

Time for candidate selection: 0.40 seconds

### Candidate
type: DSZ, layer: 5, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9785239, upper bound: 2.9786537
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9790017, upper bound: 2.9786537
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 37

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.40 seconds

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9394535, upper bound: 2.9395878
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9395878, upper bound: 2.9394535
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9802560, upper bound: 2.9805424
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9805266, upper bound: 2.9800448
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 41

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9802560, upper bound: 2.9805458
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9805243, upper bound: 2.9799909
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 13

Time for candidate selection: 0.40 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9810562, upper bound: 2.9813299
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9810562, upper bound: 2.9812307
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 1

Time for candidate selection: 0.40 seconds

### Candidate
type: DSZ, layer: 5, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9802517, upper bound: 2.9805333
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9805214, upper bound: 2.9802474
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 41

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38

Time for candidate selection: 0.40 seconds

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9789409, upper bound: 2.9789439
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9789409, upper bound: 2.9789439
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 1

Time for candidate selection: 0.40 seconds

### Candidate
type: DSZ, layer: 5, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 39

Time for candidate selection: 0.59 seconds

### Candidate
type: DSZ, layer: 5, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9802560, upper bound: 2.9807060
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9805239, upper bound: 2.9802389
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 37

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.61 seconds

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9802560, upper bound: 2.9805424
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9805266, upper bound: 2.9801436
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 0

Time for candidate selection: 0.60 seconds

### Candidate
type: DSZ, layer: 5, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9788312, upper bound: 2.9790046
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9788312, upper bound: 2.9790046
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 38

Time for candidate selection: 0.60 seconds

### Candidate
type: DSZ, layer: 5, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 2

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 2

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9802443, upper bound: 2.9805356
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9805281, upper bound: 2.9802555
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 43

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9810112, upper bound: 2.9811830
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9810923, upper bound: 2.9810918
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 43

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9810565, upper bound: 2.9811836
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811298, upper bound: 2.9810924
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 37

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 32

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9789217, upper bound: 2.9789498
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9789217, upper bound: 2.9789498
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 2

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9810508, upper bound: 2.9811565
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811269, upper bound: 2.9810804
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 32

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 45

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 2

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9810596, upper bound: 2.9811837
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811287, upper bound: 2.9810810
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 16

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9394535, upper bound: 2.9395878
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9395878, upper bound: 2.9394535
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 2

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 5, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9789409, upper bound: 2.9789439
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9789409, upper bound: 2.9789439
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 41

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9809327, upper bound: 2.9811399
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9810122, upper bound: 2.9810660
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 43

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9809327, upper bound: 2.9811730
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9810122, upper bound: 2.9810859
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 41

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 45

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9810565, upper bound: 2.9811836
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9810565, upper bound: 2.9810924
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.63 seconds

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9394535, upper bound: 2.9395878
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9395878, upper bound: 2.9394535
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 45
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 16

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### Candidate
type: DSZ, layer: 3, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 37

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 43

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 39

Time for candidate selection: 0.62 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9810596, upper bound: 2.9811837
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811287, upper bound: 2.9810810
time: 0.24 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 1.83 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9802552, upper bound: 2.9805236
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9805284, upper bound: 2.9802399
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9801981, upper bound: 2.9805258
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9805413, upper bound: 2.9802471
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9810573, upper bound: 2.9810397
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9811837, upper bound: 2.9810359
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9789822, upper bound: 2.9785239
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9789822, upper bound: 2.9785239
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9789636, upper bound: 2.9785239
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9789636, upper bound: 2.9785239
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9810927, upper bound: 2.9811306
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9811849, upper bound: 2.9810575
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9810359, upper bound: 2.9811341
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9811865, upper bound: 2.9810573
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9790046, upper bound: 2.9788312
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9790046, upper bound: 2.9788312
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9802389, upper bound: 2.9805239
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9807060, upper bound: 2.9802560
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9789511, upper bound: 2.9786698
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9789511, upper bound: 2.9786698
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9675341, upper bound: 2.9674900
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9394535, upper bound: 2.9395878
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9395878, upper bound: 2.9394535
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9810559, upper bound: 2.9811313
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9811834, upper bound: 2.9810559
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9789498, upper bound: 2.9789217
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9789498, upper bound: 2.9789217
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9394535, upper bound: 2.9395878
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9395878, upper bound: 2.9394535
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9810559, upper bound: 2.9811255
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9812511, upper bound: 2.9810559
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9810927, upper bound: 2.9811306
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9811849, upper bound: 2.9810575
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9789303, upper bound: 2.9789692
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9789303, upper bound: 2.9789692
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9394535, upper bound: 2.9395878
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9395878, upper bound: 2.9394535
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9786537, upper bound: 2.9790017
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9786537, upper bound: 2.9790017
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9394535, upper bound: 2.9395878
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9395878, upper bound: 2.9394535
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9802560, upper bound: 2.9805493
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9807267, upper bound: 2.9802561
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675341
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9789692, upper bound: 2.9789303
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9789692, upper bound: 2.9789303
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9810559, upper bound: 2.9812511
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9811255, upper bound: 2.9810807
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9785239, upper bound: 2.9786537
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9790017, upper bound: 2.9786537
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9394535, upper bound: 2.9395878
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9395878, upper bound: 2.9394535
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9802560, upper bound: 2.9805424
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9805266, upper bound: 2.9800448
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9802560, upper bound: 2.9805458
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9805243, upper bound: 2.9799909
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9810562, upper bound: 2.9813299
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9810562, upper bound: 2.9812307
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9802517, upper bound: 2.9805333
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9805214, upper bound: 2.9802474
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9789409, upper bound: 2.9789439
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9789409, upper bound: 2.9789439
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9802560, upper bound: 2.9807060
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9805239, upper bound: 2.9802389
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9802560, upper bound: 2.9805424
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9805266, upper bound: 2.9801436
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9788312, upper bound: 2.9790046
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9788312, upper bound: 2.9790046
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9802443, upper bound: 2.9805356
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9805281, upper bound: 2.9802555
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9810112, upper bound: 2.9811830
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9810923, upper bound: 2.9810918
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9810565, upper bound: 2.9811836
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9811298, upper bound: 2.9810924
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9789217, upper bound: 2.9789498
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9789217, upper bound: 2.9789498
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9810508, upper bound: 2.9811565
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9811269, upper bound: 2.9810804
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9810596, upper bound: 2.9811837
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9811287, upper bound: 2.9810810
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9394535, upper bound: 2.9395878
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9395878, upper bound: 2.9394535
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9789409, upper bound: 2.9789439
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9789409, upper bound: 2.9789439
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9809327, upper bound: 2.9811399
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9810122, upper bound: 2.9810660
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9809327, upper bound: 2.9811730
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9810122, upper bound: 2.9810859
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9810565, upper bound: 2.9811836
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9810565, upper bound: 2.9810924
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9394535, upper bound: 2.9395878
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9395878, upper bound: 2.9394535
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9810596, upper bound: 2.9811837
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.83
Output dim: 0, lower bound: -2.9811287, upper bound: 2.9810810

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9780520, upper bound: 2.9779980
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9780520, upper bound: 2.9779980
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9780520, upper bound: 2.9779980
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9780520, upper bound: 2.9779980
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9394535, upper bound: 2.9394535
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9395878, upper bound: 2.9394535
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 9

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9803109, upper bound: 2.9802133
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9805413, upper bound: 2.9802471
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9789864, upper bound: 2.9785387
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9789864, upper bound: 2.9785387
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9802153, upper bound: 2.9803931
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9805781, upper bound: 2.9802525
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 9

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367367
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367367
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9786444, upper bound: 2.9789358
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9786444, upper bound: 2.9789358
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674812, upper bound: 2.9675097
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9789439, upper bound: 2.9789290
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9789439, upper bound: 2.9789290
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9781535, upper bound: 2.9779081
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9781714, upper bound: 2.9779081
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9789864, upper bound: 2.9788312
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9790046, upper bound: 2.9788237
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9394535, upper bound: 2.9394535
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9395878, upper bound: 2.9394535
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674723, upper bound: 2.9673606
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674755, upper bound: 2.9674723
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9789761, upper bound: 2.9786698
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9789822, upper bound: 2.9786615
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9802555, upper bound: 2.9805281
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9804950, upper bound: 2.9802126
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9788144, upper bound: 2.9789877
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9788144, upper bound: 2.9789877
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9789386, upper bound: 2.9789217
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9789498, upper bound: 2.9789188
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9789386, upper bound: 2.9789217
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9789498, upper bound: 2.9789188
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9802450, upper bound: 2.9805194
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9804184, upper bound: 2.9802441
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9368765, upper bound: 2.9367367
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9368036, upper bound: 2.9367367
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9785239, upper bound: 2.9789934
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9785239, upper bound: 2.9789934
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9785239, upper bound: 2.9789871
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9785239, upper bound: 2.9789871
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 9

### Candidate
type: DSZ, layer: 5, pos: 46

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9789209, upper bound: 2.9789692
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9789303, upper bound: 2.9789631
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9777291, upper bound: 2.9780951
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9777291, upper bound: 2.9780951
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9786537, upper bound: 2.9790017
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9785589, upper bound: 2.9789877
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9394535, upper bound: 2.9394535
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9395878, upper bound: 2.9394535
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9804230, upper bound: 2.9802554
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9807267, upper bound: 2.9802561
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9789631, upper bound: 2.9789303
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9789692, upper bound: 2.9789209
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9780464, upper bound: 2.9780237
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9780464, upper bound: 2.9780237
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9802560, upper bound: 2.9806479
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9804173, upper bound: 2.9802140
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9802441, upper bound: 2.9804184
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9805194, upper bound: 2.9802450
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 9

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9780951, upper bound: 2.9777291
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9780951, upper bound: 2.9777291
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9780951, upper bound: 2.9777291
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9780951, upper bound: 2.9777291
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674723, upper bound: 2.9674755
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9673606, upper bound: 2.9674760
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9780700, upper bound: 2.9776384
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9780700, upper bound: 2.9776384
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674723, upper bound: 2.9674755
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9673606, upper bound: 2.9674760
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9804182, upper bound: 2.9799908
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9805243, upper bound: 2.9799909
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9788209, upper bound: 2.9789636
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9788209, upper bound: 2.9789636
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675097, upper bound: 2.9674812
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675341
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9777733, upper bound: 2.9780600
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9777733, upper bound: 2.9780600
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 13

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9394535, upper bound: 2.9395878
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9394535, upper bound: 2.9394535
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9780247, upper bound: 2.9780192
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9780247, upper bound: 2.9780192
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674723, upper bound: 2.9674755
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9673606, upper bound: 2.9674760
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9804100, upper bound: 2.9802114
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9805239, upper bound: 2.9802389
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674723, upper bound: 2.9674755
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9673606, upper bound: 2.9674760
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9804108, upper bound: 2.9801331
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9805266, upper bound: 2.9801436
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9788237, upper bound: 2.9790046
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9788312, upper bound: 2.9789864
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9780951, upper bound: 2.9778959
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9780951, upper bound: 2.9778959
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674760, upper bound: 2.9673606
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9673606, upper bound: 2.9674723
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9789549, upper bound: 2.9789303
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9789549, upper bound: 2.9789303
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367383
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367367
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367367
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367367
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367367
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367367
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9789188, upper bound: 2.9789498
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9789217, upper bound: 2.9789386
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9779970, upper bound: 2.9780848
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9779970, upper bound: 2.9780792
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367367
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367367
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 9

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9788494, upper bound: 2.9788080
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9788494, upper bound: 2.9788080
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367600
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9368252
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9802144, upper bound: 2.9804236
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9805241, upper bound: 2.9802545
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674812, upper bound: 2.9675097
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367367
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367367
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674812, upper bound: 2.9675097
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9675097, upper bound: 2.9674812
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367367
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367367
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9789457, upper bound: 2.9786617
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9789457, upper bound: 2.9786617
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367921
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9369508
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367921
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9369508
time: 0.23 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 1.32 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9780520, upper bound: 2.9779980
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9780520, upper bound: 2.9779980
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9780520, upper bound: 2.9779980
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9780520, upper bound: 2.9779980
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9394535, upper bound: 2.9394535
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9395878, upper bound: 2.9394535
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9803109, upper bound: 2.9802133
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9805413, upper bound: 2.9802471
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9789864, upper bound: 2.9785387
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9789864, upper bound: 2.9785387
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9802153, upper bound: 2.9803931
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9805781, upper bound: 2.9802525
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367367
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367367
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9786444, upper bound: 2.9789358
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9786444, upper bound: 2.9789358
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9674812, upper bound: 2.9675097
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9789439, upper bound: 2.9789290
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9789439, upper bound: 2.9789290
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9781535, upper bound: 2.9779081
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9781714, upper bound: 2.9779081
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9789864, upper bound: 2.9788312
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9790046, upper bound: 2.9788237
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9394535, upper bound: 2.9394535
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9395878, upper bound: 2.9394535
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9674723, upper bound: 2.9673606
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9674755, upper bound: 2.9674723
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9789761, upper bound: 2.9786698
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9789822, upper bound: 2.9786615
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9802555, upper bound: 2.9805281
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9804950, upper bound: 2.9802126
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9788144, upper bound: 2.9789877
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9788144, upper bound: 2.9789877
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9789386, upper bound: 2.9789217
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9789498, upper bound: 2.9789188
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9789386, upper bound: 2.9789217
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9789498, upper bound: 2.9789188
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9802450, upper bound: 2.9805194
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9804184, upper bound: 2.9802441
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9368765, upper bound: 2.9367367
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9368036, upper bound: 2.9367367
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9785239, upper bound: 2.9789934
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9785239, upper bound: 2.9789934
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9785239, upper bound: 2.9789871
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9785239, upper bound: 2.9789871
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9789209, upper bound: 2.9789692
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9789303, upper bound: 2.9789631
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9777291, upper bound: 2.9780951
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9777291, upper bound: 2.9780951
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9786537, upper bound: 2.9790017
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9785589, upper bound: 2.9789877
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9394535, upper bound: 2.9394535
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9395878, upper bound: 2.9394535
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9804230, upper bound: 2.9802554
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9807267, upper bound: 2.9802561
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9789631, upper bound: 2.9789303
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9789692, upper bound: 2.9789209
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9780464, upper bound: 2.9780237
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9780464, upper bound: 2.9780237
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9802560, upper bound: 2.9806479
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9804173, upper bound: 2.9802140
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9802441, upper bound: 2.9804184
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9805194, upper bound: 2.9802450
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9780951, upper bound: 2.9777291
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9780951, upper bound: 2.9777291
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9780951, upper bound: 2.9777291
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9780951, upper bound: 2.9777291
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9674723, upper bound: 2.9674755
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9673606, upper bound: 2.9674760
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9780700, upper bound: 2.9776384
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9780700, upper bound: 2.9776384
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9674723, upper bound: 2.9674755
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9673606, upper bound: 2.9674760
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9804182, upper bound: 2.9799908
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9805243, upper bound: 2.9799909
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9788209, upper bound: 2.9789636
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9788209, upper bound: 2.9789636
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9675097, upper bound: 2.9674812
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675341
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9777733, upper bound: 2.9780600
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9777733, upper bound: 2.9780600
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9394535, upper bound: 2.9395878
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9394535, upper bound: 2.9394535
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9780247, upper bound: 2.9780192
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9780247, upper bound: 2.9780192
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9674723, upper bound: 2.9674755
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9673606, upper bound: 2.9674760
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9804100, upper bound: 2.9802114
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9805239, upper bound: 2.9802389
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9674723, upper bound: 2.9674755
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9673606, upper bound: 2.9674760
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9804108, upper bound: 2.9801331
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9805266, upper bound: 2.9801436
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9788237, upper bound: 2.9790046
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9788312, upper bound: 2.9789864
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9780951, upper bound: 2.9778959
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9780951, upper bound: 2.9778959
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9674760, upper bound: 2.9673606
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9673606, upper bound: 2.9674723
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9789549, upper bound: 2.9789303
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9789549, upper bound: 2.9789303
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367383
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367367
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367367
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367367
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367367
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367367
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9789188, upper bound: 2.9789498
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9789217, upper bound: 2.9789386
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9779970, upper bound: 2.9780848
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9779970, upper bound: 2.9780792
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367367
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367367
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9788494, upper bound: 2.9788080
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9788494, upper bound: 2.9788080
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367600
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9368252
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9802144, upper bound: 2.9804236
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9805241, upper bound: 2.9802545
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674900
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9674812, upper bound: 2.9675097
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367367
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367367
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9675322, upper bound: 2.9674901
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9674812, upper bound: 2.9675097
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9675097, upper bound: 2.9674812
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9674901, upper bound: 2.9675322
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367367
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367367
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9789457, upper bound: 2.9786617
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9789457, upper bound: 2.9786617
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367921
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9369508
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367921
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.32
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9369508

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 9

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367367
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367367
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9673606, upper bound: 2.9673606
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674755, upper bound: 2.9674723
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9779773, upper bound: 2.9776907
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9779221, upper bound: 2.9776907
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9779773, upper bound: 2.9776907
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9779719, upper bound: 2.9776907
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9780792, upper bound: 2.9776662
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9781516, upper bound: 2.9776662
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9781712, upper bound: 2.9776662
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9781712, upper bound: 2.9776662
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9777669, upper bound: 2.9778601
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9777669, upper bound: 2.9778874
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9780192, upper bound: 2.9778578
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9780192, upper bound: 2.9778850
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 1

### Candidate
type: DSZ, layer: 5, pos: 46

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9779773, upper bound: 2.9779081
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9779719, upper bound: 2.9779081
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9778969, upper bound: 2.9777733
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9778686, upper bound: 2.9777733
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9780600, upper bound: 2.9776999
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9780600, upper bound: 2.9776999
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9368037
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9368874
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367367
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367367
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9778959, upper bound: 2.9779079
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9778959, upper bound: 2.9779385
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9778959, upper bound: 2.9779079
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9777699, upper bound: 2.9779385
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 13

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9779347, upper bound: 2.9779970
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9779221, upper bound: 2.9779970
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 13

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9780792, upper bound: 2.9778358
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9780848, upper bound: 2.9778617
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 1

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9780792, upper bound: 2.9778358
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9780848, upper bound: 2.9778617
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674723, upper bound: 2.9674755
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9673606, upper bound: 2.9674760
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674723, upper bound: 2.9673606
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9673606, upper bound: 2.9673606
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9776384, upper bound: 2.9780700
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9776105, upper bound: 2.9780700
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9776105, upper bound: 2.9778878
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9776105, upper bound: 2.9779193
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9778889, upper bound: 2.9780464
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9778617, upper bound: 2.9780464
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9780237, upper bound: 2.9778642
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9780237, upper bound: 2.9778944
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9776971, upper bound: 2.9780951
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9777291, upper bound: 2.9779079
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9776971, upper bound: 2.9780951
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9777291, upper bound: 2.9779385
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9776971, upper bound: 2.9780951
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9776971, upper bound: 2.9780951
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9778669, upper bound: 2.9780914
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9778669, upper bound: 2.9780914
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 13

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9780343, upper bound: 2.9779328
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9780343, upper bound: 2.9779328
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9780464, upper bound: 2.9778617
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9780464, upper bound: 2.9778889
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Candidate
type: DSZ, layer: 5, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9778730, upper bound: 2.9780848
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9778730, upper bound: 2.9780848
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367367
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367367
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Candidate
type: DSZ, layer: 5, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9368036
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9368765
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9673606, upper bound: 2.9673606
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9673606, upper bound: 2.9674723
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9779385, upper bound: 2.9777291
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9780951, upper bound: 2.9776971
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9779079, upper bound: 2.9777291
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9780951, upper bound: 2.9776971
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9779385, upper bound: 2.9777291
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9780951, upper bound: 2.9776971
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9778787, upper bound: 2.9776105
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9778787, upper bound: 2.9776105
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9780424, upper bound: 2.9776144
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9780424, upper bound: 2.9776144
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9777814, upper bound: 2.9780400
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9777814, upper bound: 2.9780400
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674747, upper bound: 2.9673606
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9673606, upper bound: 2.9673606
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367367
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367367
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9674747, upper bound: 2.9673606
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9673606, upper bound: 2.9673606
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9780273, upper bound: 2.9777277
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9780273, upper bound: 2.9777277
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 9

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9779081, upper bound: 2.9779719
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9779081, upper bound: 2.9779773
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9779385, upper bound: 2.9778959
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9780951, upper bound: 2.9777699
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9778864, upper bound: 2.9780237
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9778572, upper bound: 2.9780237
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Candidate
type: DSZ, layer: 5, pos: 9

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 1

### Candidate
type: DSZ, layer: 5, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 9

### Candidate
type: DSZ, layer: 5, pos: 46

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 9

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9779493, upper bound: 2.9777725
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9779493, upper bound: 2.9777725
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Candidate
type: DSZ, layer: 5, pos: 9

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367600
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9368252
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367367
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367367
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9780273, upper bound: 2.9777277
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9780273, upper bound: 2.9777277
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Candidate
type: DSZ, layer: 5, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9780273, upper bound: 2.9777277
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9780273, upper bound: 2.9777277
time: 0.24 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 1.45 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367367
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367367
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9673606, upper bound: 2.9673606
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9674755, upper bound: 2.9674723
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9779773, upper bound: 2.9776907
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9779221, upper bound: 2.9776907
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9779773, upper bound: 2.9776907
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9779719, upper bound: 2.9776907
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9780792, upper bound: 2.9776662
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9781516, upper bound: 2.9776662
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9781712, upper bound: 2.9776662
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9781712, upper bound: 2.9776662
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9777669, upper bound: 2.9778601
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9777669, upper bound: 2.9778874
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9780192, upper bound: 2.9778578
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9780192, upper bound: 2.9778850
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9779773, upper bound: 2.9779081
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9779719, upper bound: 2.9779081
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9778969, upper bound: 2.9777733
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9778686, upper bound: 2.9777733
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9780600, upper bound: 2.9776999
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9780600, upper bound: 2.9776999
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9368037
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9368874
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367367
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367367
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9778959, upper bound: 2.9779079
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9778959, upper bound: 2.9779385
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9778959, upper bound: 2.9779079
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9777699, upper bound: 2.9779385
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9779347, upper bound: 2.9779970
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9779221, upper bound: 2.9779970
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9780792, upper bound: 2.9778358
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9780848, upper bound: 2.9778617
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9780792, upper bound: 2.9778358
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9780848, upper bound: 2.9778617
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9674723, upper bound: 2.9674755
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9673606, upper bound: 2.9674760
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9674723, upper bound: 2.9673606
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9673606, upper bound: 2.9673606
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9776384, upper bound: 2.9780700
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9776105, upper bound: 2.9780700
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9776105, upper bound: 2.9778878
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9776105, upper bound: 2.9779193
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9778889, upper bound: 2.9780464
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9778617, upper bound: 2.9780464
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9780237, upper bound: 2.9778642
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9780237, upper bound: 2.9778944
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9776971, upper bound: 2.9780951
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9777291, upper bound: 2.9779079
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9776971, upper bound: 2.9780951
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9777291, upper bound: 2.9779385
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9776971, upper bound: 2.9780951
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9776971, upper bound: 2.9780951
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9778669, upper bound: 2.9780914
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9778669, upper bound: 2.9780914
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9780343, upper bound: 2.9779328
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9780343, upper bound: 2.9779328
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9780464, upper bound: 2.9778617
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9780464, upper bound: 2.9778889
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9778730, upper bound: 2.9780848
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9778730, upper bound: 2.9780848
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367367
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367367
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9368036
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9368765
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9673606, upper bound: 2.9673606
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9673606, upper bound: 2.9674723
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9779385, upper bound: 2.9777291
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9780951, upper bound: 2.9776971
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9779079, upper bound: 2.9777291
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9780951, upper bound: 2.9776971
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9779385, upper bound: 2.9777291
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9780951, upper bound: 2.9776971
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9778787, upper bound: 2.9776105
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9778787, upper bound: 2.9776105
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9780424, upper bound: 2.9776144
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9780424, upper bound: 2.9776144
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9777814, upper bound: 2.9780400
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9777814, upper bound: 2.9780400
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9674747, upper bound: 2.9673606
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9673606, upper bound: 2.9673606
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367367
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367367
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9674747, upper bound: 2.9673606
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9673606, upper bound: 2.9673606
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9780273, upper bound: 2.9777277
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9780273, upper bound: 2.9777277
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9779081, upper bound: 2.9779719
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9779081, upper bound: 2.9779773
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9779385, upper bound: 2.9778959
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9780951, upper bound: 2.9777699
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9778864, upper bound: 2.9780237
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9778572, upper bound: 2.9780237
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9644994, upper bound: 2.9644994
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9779493, upper bound: 2.9777725
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9779493, upper bound: 2.9777725
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367600
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9368252
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367367
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9367367, upper bound: 2.9367367
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9780273, upper bound: 2.9777277
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9780273, upper bound: 2.9777277
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9780273, upper bound: 2.9777277
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.45
Output dim: 0, lower bound: -2.9780273, upper bound: 2.9777277

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 1

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 13

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 1

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 9

### Candidate
type: DSZ, layer: 5, pos: 1

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Candidate
type: DSZ, layer: 5, pos: 1

### Candidate
type: DSZ, layer: 5, pos: 39

### Candidate
type: DSZ, layer: 5, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 9

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 9

### Candidate
type: DSZ, layer: 5, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 39

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Candidate
type: DSZ, layer: 5, pos: 13

### Candidate
type: DSZ, layer: 5, pos: 46

### Candidate
type: DSZ, layer: 5, pos: 38

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 46

### Candidate
type: DSZ, layer: 5, pos: 13

### Candidate
type: DSZ, layer: 5, pos: 0

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 13

### Candidate
type: DSZ, layer: 5, pos: 9

### Candidate
type: DSZ, layer: 5, pos: 38

### Candidate
type: DSZ, layer: 5, pos: 46

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Candidate
type: DSZ, layer: 5, pos: 46

### Candidate
type: DSZ, layer: 5, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 38

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 1

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 38

### Candidate
type: DSZ, layer: 5, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 46
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 39

### Candidate
type: DSZ, layer: 5, pos: 1

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314
1: -1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077
2: -1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894
3: -2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013
4: -1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 38
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 9
type: DSZ, layer: 5, pos: 39
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 1

### Candidate
type: DSZ, layer: 5, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 38

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
time: 0.24 seconds

## Summary of splitting (split count: 9)
- Time for DS candidates: 1.72 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.72
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.72
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.72
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.72
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.72
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.72
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.72
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.72
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.72
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.72
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.72
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.72
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.72
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.72
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.72
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.72
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.72
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.72
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.72
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.72
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.72
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.72
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.72
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.72
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.72
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.72
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.72
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.72
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.72
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.72
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.72
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.72
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.72
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.72
Output dim: 0, lower bound: -2.9643570, upper bound: 2.9643570

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 1.41 + 414.56 = 415.97 seconds
