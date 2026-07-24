## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_8.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 4)
Time budget: 420 seconds
Split limit: 100
Threshold: 53.8414279856


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954)
1: (-20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342)
2: (-34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566)
3: (-40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698)
4: (-30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.98 + 1.42 = 2.41 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -54.3304016, upper bound: 54.3304016

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.3229415, upper bound: 54.3229415
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.3229415, upper bound: 54.3232450
time: 0.41 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.83 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.83
Output dim: 4, lower bound: -54.3229415, upper bound: 54.3229415
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.83
Output dim: 4, lower bound: -54.3229415, upper bound: 54.3232450

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.3168664, upper bound: 54.3168648
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.3168648, upper bound: 54.3168664
time: 0.38 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.3220429, upper bound: 54.3220524
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.3220410, upper bound: 54.3224197
time: 0.37 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.66 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.66
Output dim: 4, lower bound: -54.3168664, upper bound: 54.3168648
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.66
Output dim: 4, lower bound: -54.3168648, upper bound: 54.3168664
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.66
Output dim: 4, lower bound: -54.3220429, upper bound: 54.3220524
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.66
Output dim: 4, lower bound: -54.3220410, upper bound: 54.3224197

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1264146, upper bound: 54.1264146
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1264146, upper bound: 54.1264146
time: 0.37 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1264146, upper bound: 54.1264146
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1264146, upper bound: 54.1264146
time: 0.35 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.3159408, upper bound: 54.3159559
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.3159390, upper bound: 54.3159405
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.3217196, upper bound: 54.3219279
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.3217196, upper bound: 54.3217903
time: 0.37 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 1.67 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.67
Output dim: 4, lower bound: -54.1264146, upper bound: 54.1264146
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.67
Output dim: 4, lower bound: -54.1264146, upper bound: 54.1264146
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.67
Output dim: 4, lower bound: -54.1264146, upper bound: 54.1264146
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.67
Output dim: 4, lower bound: -54.1264146, upper bound: 54.1264146
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.67
Output dim: 4, lower bound: -54.3159408, upper bound: 54.3159559
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.67
Output dim: 4, lower bound: -54.3159390, upper bound: 54.3159405
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.67
Output dim: 4, lower bound: -54.3217196, upper bound: 54.3219279
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.67
Output dim: 4, lower bound: -54.3217196, upper bound: 54.3217903

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.8176893, upper bound: 53.8176893
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.8176893, upper bound: 53.8176893
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
time: 0.35 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1264146, upper bound: 54.1264146
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1264146, upper bound: 54.1264146
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1264146, upper bound: 54.1264145
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1264146, upper bound: 54.1264146
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1806568, upper bound: 54.1806845
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1806641, upper bound: 54.1806568
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.0808831, upper bound: 54.0808831
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.0808831, upper bound: 54.0808831
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1840539, upper bound: 54.1841199
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1840539, upper bound: 54.1840539
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1840539, upper bound: 54.1840539
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1840579, upper bound: 54.1840539
time: 0.41 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 1.89 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 1.89
Output dim: 4, lower bound: -53.8176893, upper bound: 53.8176893
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 1.89
Output dim: 4, lower bound: -53.8176893, upper bound: 53.8176893
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.89
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.89
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.89
Output dim: 4, lower bound: -54.1264146, upper bound: 54.1264146
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.89
Output dim: 4, lower bound: -54.1264146, upper bound: 54.1264146
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.89
Output dim: 4, lower bound: -54.1264146, upper bound: 54.1264145
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.89
Output dim: 4, lower bound: -54.1264146, upper bound: 54.1264146
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.89
Output dim: 4, lower bound: -54.1806568, upper bound: 54.1806845
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.89
Output dim: 4, lower bound: -54.1806641, upper bound: 54.1806568
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.89
Output dim: 4, lower bound: -54.0808831, upper bound: 54.0808831
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.89
Output dim: 4, lower bound: -54.0808831, upper bound: 54.0808831
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.89
Output dim: 4, lower bound: -54.1840539, upper bound: 54.1841199
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.89
Output dim: 4, lower bound: -54.1840539, upper bound: 54.1840539
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.89
Output dim: 4, lower bound: -54.1840539, upper bound: 54.1840539
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.89
Output dim: 4, lower bound: -54.1840579, upper bound: 54.1840539

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
time: 0.37 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.8176893, upper bound: 53.8176893
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.8176893, upper bound: 53.8176893
time: 0.37 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1264145, upper bound: 54.1264146
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1264145, upper bound: 54.1264146
time: 0.36 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1264146, upper bound: 54.1264146
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1264146, upper bound: 54.1264146
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9783830, upper bound: 53.9784386
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9783830, upper bound: 53.9784386
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9784350, upper bound: 53.9784064
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9783830, upper bound: 53.9784064
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.0739595, upper bound: 54.0739595
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.0739595, upper bound: 54.0739595
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1265705, upper bound: 54.1266424
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1265705, upper bound: 54.1266423
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1840539, upper bound: 54.1840539
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1840539, upper bound: 54.1840539
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1840539, upper bound: 54.1840539
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1840539, upper bound: 54.1840539
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1265705, upper bound: 54.1265705
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1265705, upper bound: 54.1265705
time: 0.41 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 1.81 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.81
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.81
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.81
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.81
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.81
Output dim: 4, lower bound: -53.8176893, upper bound: 53.8176893
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.81
Output dim: 4, lower bound: -53.8176893, upper bound: 53.8176893
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.81
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.81
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.81
Output dim: 4, lower bound: -54.1264145, upper bound: 54.1264146
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.81
Output dim: 4, lower bound: -54.1264145, upper bound: 54.1264146
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.81
Output dim: 4, lower bound: -54.1264146, upper bound: 54.1264146
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.81
Output dim: 4, lower bound: -54.1264146, upper bound: 54.1264146
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.81
Output dim: 4, lower bound: -53.9783830, upper bound: 53.9784386
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.81
Output dim: 4, lower bound: -53.9783830, upper bound: 53.9784386
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.81
Output dim: 4, lower bound: -53.9784350, upper bound: 53.9784064
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.81
Output dim: 4, lower bound: -53.9783830, upper bound: 53.9784064
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.81
Output dim: 4, lower bound: -54.0739595, upper bound: 54.0739595
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.81
Output dim: 4, lower bound: -54.0739595, upper bound: 54.0739595
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.81
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.81
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.81
Output dim: 4, lower bound: -54.1265705, upper bound: 54.1266424
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.81
Output dim: 4, lower bound: -54.1265705, upper bound: 54.1266423
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.81
Output dim: 4, lower bound: -54.1840539, upper bound: 54.1840539
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.81
Output dim: 4, lower bound: -54.1840539, upper bound: 54.1840539
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.81
Output dim: 4, lower bound: -54.1840539, upper bound: 54.1840539
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.81
Output dim: 4, lower bound: -54.1840539, upper bound: 54.1840539
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.81
Output dim: 4, lower bound: -54.1265705, upper bound: 54.1265705
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.81
Output dim: 4, lower bound: -54.1265705, upper bound: 54.1265705

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.8159489, upper bound: 53.8159489
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.8159489, upper bound: 53.8159489
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1252472, upper bound: 54.1252473
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.8062985, upper bound: 53.8062985
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.8062985, upper bound: 53.8062985
time: 0.35 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.8062985, upper bound: 53.8062985
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.8062985, upper bound: 53.8062985
time: 0.34 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.8176893, upper bound: 53.8176893
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.8176893, upper bound: 53.8176893
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1264146, upper bound: 54.1264146
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1264146, upper bound: 54.1264146
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9783292, upper bound: 53.9784344
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9783537, upper bound: 53.9783771
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151851
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151851
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9783830, upper bound: 53.9784063
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9784350, upper bound: 53.9783829
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1265705, upper bound: 54.1265705
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1265705, upper bound: 54.1266424
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252938
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1806568, upper bound: 54.1806566
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1806566, upper bound: 54.1806566
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1806566, upper bound: 54.1806566
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1806566, upper bound: 54.1806566
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.8159489, upper bound: 53.8159489
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.8159489, upper bound: 53.8159489
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9216404, upper bound: 53.9216404
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9216404, upper bound: 53.9216404
time: 0.38 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.08 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.08
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.08
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.08
Output dim: 4, lower bound: -53.8159489, upper bound: 53.8159489
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.08
Output dim: 4, lower bound: -53.8159489, upper bound: 53.8159489
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.08
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.08
Output dim: 4, lower bound: -54.1252472, upper bound: 54.1252473
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.08
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.08
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.08
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.08
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.08
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.08
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.08
Output dim: 4, lower bound: -53.8062985, upper bound: 53.8062985
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.08
Output dim: 4, lower bound: -53.8062985, upper bound: 53.8062985
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.08
Output dim: 4, lower bound: -53.8062985, upper bound: 53.8062985
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.08
Output dim: 4, lower bound: -53.8062985, upper bound: 53.8062985
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.08
Output dim: 4, lower bound: -53.8176893, upper bound: 53.8176893
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.08
Output dim: 4, lower bound: -53.8176893, upper bound: 53.8176893
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.08
Output dim: 4, lower bound: -54.1264146, upper bound: 54.1264146
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.08
Output dim: 4, lower bound: -54.1264146, upper bound: 54.1264146
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.08
Output dim: 4, lower bound: -53.9783292, upper bound: 53.9784344
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.08
Output dim: 4, lower bound: -53.9783537, upper bound: 53.9783771
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.08
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151851
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.08
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151851
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.08
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.08
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.08
Output dim: 4, lower bound: -53.9783830, upper bound: 53.9784063
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.08
Output dim: 4, lower bound: -53.9784350, upper bound: 53.9783829
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.08
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.08
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.08
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.08
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.08
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.08
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.08
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.08
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.08
Output dim: 4, lower bound: -54.1265705, upper bound: 54.1265705
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.08
Output dim: 4, lower bound: -54.1265705, upper bound: 54.1266424
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.08
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252938
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.08
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.08
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.08
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.08
Output dim: 4, lower bound: -54.1806568, upper bound: 54.1806566
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.08
Output dim: 4, lower bound: -54.1806566, upper bound: 54.1806566
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.08
Output dim: 4, lower bound: -54.1806566, upper bound: 54.1806566
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.08
Output dim: 4, lower bound: -54.1806566, upper bound: 54.1806566
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.08
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.08
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.08
Output dim: 4, lower bound: -53.8159489, upper bound: 53.8159489
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.08
Output dim: 4, lower bound: -53.8159489, upper bound: 53.8159489
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.08
Output dim: 4, lower bound: -53.9216404, upper bound: 53.9216404
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.08
Output dim: 4, lower bound: -53.9216404, upper bound: 53.9216404

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
time: 0.35 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
time: 0.36 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 11

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9162499, upper bound: 53.9162499
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9162499, upper bound: 53.9162499
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.8062985, upper bound: 53.8062985
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.8062985, upper bound: 53.8062985
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9152116
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9152116
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252472
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9783292, upper bound: 53.9783292
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9783292, upper bound: 53.9783292
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9216404, upper bound: 53.9216404
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9216404, upper bound: 53.9216404
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.40 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 2.25 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.25
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.25
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.25
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.25
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.25
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.25
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.25
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.25
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.25
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.25
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.25
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.25
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.25
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.25
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.25
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.25
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.25
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.25
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.25
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.25
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.25
Output dim: 4, lower bound: -53.9162499, upper bound: 53.9162499
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.25
Output dim: 4, lower bound: -53.9162499, upper bound: 53.9162499
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.25
Output dim: 4, lower bound: -53.8062985, upper bound: 53.8062985
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.25
Output dim: 4, lower bound: -53.8062985, upper bound: 53.8062985
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.25
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.25
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.25
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.25
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.25
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.25
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.25
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.25
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.25
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.25
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.25
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.25
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.25
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.25
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.25
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.25
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.25
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.25
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.25
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.25
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.25
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.25
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.25
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.25
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.25
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9152116
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.25
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9152116
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.25
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.25
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.25
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.25
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.25
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252472
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.25
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.25
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.25
Output dim: 4, lower bound: -54.1252473, upper bound: 54.1252473
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.25
Output dim: 4, lower bound: -53.9783292, upper bound: 53.9783292
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.25
Output dim: 4, lower bound: -53.9783292, upper bound: 53.9783292
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.25
Output dim: 4, lower bound: -53.9216404, upper bound: 53.9216404
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.25
Output dim: 4, lower bound: -53.9216404, upper bound: 53.9216404
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.25
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.25
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
time: 0.34 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
time: 0.35 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 11

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 11

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 11

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 11

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151851, upper bound: 53.9151821
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9152116
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 11

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
time: 0.42 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 2.06 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.9151851, upper bound: 53.9151821
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9152116
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.8026388, upper bound: 53.8026388
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.7836169, upper bound: 53.7836169
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.06
Output dim: 4, lower bound: -53.9151821, upper bound: 53.9151821

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 11

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.37 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 11

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 11

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 11

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.34 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 11

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.35 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 11

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.35 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.34 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 11

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954
1: -20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342
2: -34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566
3: -40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698
4: -30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
time: 0.40 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 2.01 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.01
Output dim: 4, lower bound: -53.7715942, upper bound: 53.7715942

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 2.41 + 328.84 = 331.25 seconds
