## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 2)
Time budget: 420 seconds
Split limit: 100
Threshold: 0.07289891999999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720)
1: (-0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452)
2: (0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376)
3: (-0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086)
4: (0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.77 + 0.65 = 2.42 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0809988, upper bound: 0.0809988

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0807117, upper bound: 0.0809970
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0809970, upper bound: 0.0807117
time: 0.22 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.60 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.60
Output dim: 0, lower bound: -0.0807117, upper bound: 0.0809970
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.60
Output dim: 0, lower bound: -0.0809970, upper bound: 0.0807117

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0780590, upper bound: 0.0784160
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0781955, upper bound: 0.0781469
time: 0.22 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0781469, upper bound: 0.0781955
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0784160, upper bound: 0.0780590
time: 0.22 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.17 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.17
Output dim: 0, lower bound: -0.0780590, upper bound: 0.0784160
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.17
Output dim: 0, lower bound: -0.0781955, upper bound: 0.0781469
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.17
Output dim: 0, lower bound: -0.0781469, upper bound: 0.0781955
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.17
Output dim: 0, lower bound: -0.0784160, upper bound: 0.0780590

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0780186, upper bound: 0.0782362
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0780186, upper bound: 0.0782362
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0781574, upper bound: 0.0780385
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0781574, upper bound: 0.0780385
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0780385, upper bound: 0.0781574
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0780385, upper bound: 0.0781574
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0782362, upper bound: 0.0780186
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0782362, upper bound: 0.0780186
time: 0.21 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.84 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.84
Output dim: 0, lower bound: -0.0780186, upper bound: 0.0782362
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.84
Output dim: 0, lower bound: -0.0780186, upper bound: 0.0782362
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.84
Output dim: 0, lower bound: -0.0781574, upper bound: 0.0780385
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.84
Output dim: 0, lower bound: -0.0781574, upper bound: 0.0780385
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.84
Output dim: 0, lower bound: -0.0780385, upper bound: 0.0781574
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.84
Output dim: 0, lower bound: -0.0780385, upper bound: 0.0781574
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.84
Output dim: 0, lower bound: -0.0782362, upper bound: 0.0780186
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.84
Output dim: 0, lower bound: -0.0782362, upper bound: 0.0780186

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Candidate
type: DSZ, layer: 1, pos: 8

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0777385, upper bound: 0.0782362
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0777385, upper bound: 0.0781858
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Candidate
type: DSZ, layer: 1, pos: 8

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0778873, upper bound: 0.0782362
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0779385, upper bound: 0.0781858
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Candidate
type: DSZ, layer: 1, pos: 8

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0779179, upper bound: 0.0779468
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0778709, upper bound: 0.0778820
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Candidate
type: DSZ, layer: 1, pos: 8

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0780630, upper bound: 0.0778912
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0781043, upper bound: 0.0778330
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Candidate
type: DSZ, layer: 1, pos: 8

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0778330, upper bound: 0.0781043
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0778912, upper bound: 0.0780630
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Candidate
type: DSZ, layer: 1, pos: 8

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0778820, upper bound: 0.0778709
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0779468, upper bound: 0.0779179
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Candidate
type: DSZ, layer: 1, pos: 8

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0781858, upper bound: 0.0779385
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0782362, upper bound: 0.0778873
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Candidate
type: DSZ, layer: 1, pos: 8

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0781858, upper bound: 0.0777385
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0782362, upper bound: 0.0777385
time: 0.22 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 3.10 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 0, lower bound: -0.0777385, upper bound: 0.0782362
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 0, lower bound: -0.0777385, upper bound: 0.0781858
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 0, lower bound: -0.0778873, upper bound: 0.0782362
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 0, lower bound: -0.0779385, upper bound: 0.0781858
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 0, lower bound: -0.0779179, upper bound: 0.0779468
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 0, lower bound: -0.0778709, upper bound: 0.0778820
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 0, lower bound: -0.0780630, upper bound: 0.0778912
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 0, lower bound: -0.0781043, upper bound: 0.0778330
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 0, lower bound: -0.0778330, upper bound: 0.0781043
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 0, lower bound: -0.0778912, upper bound: 0.0780630
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 0, lower bound: -0.0778820, upper bound: 0.0778709
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 0, lower bound: -0.0779468, upper bound: 0.0779179
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 0, lower bound: -0.0781858, upper bound: 0.0779385
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 0, lower bound: -0.0782362, upper bound: 0.0778873
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 0, lower bound: -0.0781858, upper bound: 0.0777385
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 0, lower bound: -0.0782362, upper bound: 0.0777385

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0774122, upper bound: 0.0780396
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0774219, upper bound: 0.0781463
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0774192, upper bound: 0.0780327
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0774285, upper bound: 0.0780978
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0775784, upper bound: 0.0779211
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0777081, upper bound: 0.0781463
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0778231, upper bound: 0.0779548
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0778211, upper bound: 0.0780978
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0777037, upper bound: 0.0776309
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0778651, upper bound: 0.0778050
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0775696, upper bound: 0.0775776
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0777408, upper bound: 0.0775748
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0779711, upper bound: 0.0774342
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0780170, upper bound: 0.0775765
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0780603, upper bound: 0.0774320
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0780400, upper bound: 0.0775103
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.66 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0775103, upper bound: 0.0780400
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0774320, upper bound: 0.0780603
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.66 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0775765, upper bound: 0.0780170
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0774342, upper bound: 0.0779711
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.66 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0775748, upper bound: 0.0777408
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0775776, upper bound: 0.0775696
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.67 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0778050, upper bound: 0.0778651
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0776309, upper bound: 0.0777037
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.67 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0780978, upper bound: 0.0778211
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0779548, upper bound: 0.0778231
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.67 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0781463, upper bound: 0.0777081
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0779211, upper bound: 0.0775784
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.68 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0780978, upper bound: 0.0774285
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0780327, upper bound: 0.0774192
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.69 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0781463, upper bound: 0.0774219
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0780396, upper bound: 0.0774122
time: 0.23 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 2.31 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0774122, upper bound: 0.0780396
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0774219, upper bound: 0.0781463
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0774192, upper bound: 0.0780327
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0774285, upper bound: 0.0780978
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0775784, upper bound: 0.0779211
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0777081, upper bound: 0.0781463
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0778231, upper bound: 0.0779548
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0778211, upper bound: 0.0780978
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0777037, upper bound: 0.0776309
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0778651, upper bound: 0.0778050
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0775696, upper bound: 0.0775776
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0777408, upper bound: 0.0775748
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0779711, upper bound: 0.0774342
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0780170, upper bound: 0.0775765
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0780603, upper bound: 0.0774320
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0780400, upper bound: 0.0775103
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0775103, upper bound: 0.0780400
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0774320, upper bound: 0.0780603
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0775765, upper bound: 0.0780170
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0774342, upper bound: 0.0779711
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0775748, upper bound: 0.0777408
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0775776, upper bound: 0.0775696
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0778050, upper bound: 0.0778651
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0776309, upper bound: 0.0777037
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0780978, upper bound: 0.0778211
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0779548, upper bound: 0.0778231
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0781463, upper bound: 0.0777081
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0779211, upper bound: 0.0775784
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0780978, upper bound: 0.0774285
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0780327, upper bound: 0.0774192
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0781463, upper bound: 0.0774219
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0780396, upper bound: 0.0774122

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.66 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 21

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.65 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0752894, upper bound: 0.0756043
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0752894, upper bound: 0.0756043
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.66 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 21

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.65 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0752888, upper bound: 0.0756241
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0752888, upper bound: 0.0756241
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.66 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 21

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.65 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0752894, upper bound: 0.0755844
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0752894, upper bound: 0.0755844
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.67 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 21

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.65 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0752888, upper bound: 0.0756007
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0752888, upper bound: 0.0756007
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.67 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 21

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.65 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755079, upper bound: 0.0753806
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755079, upper bound: 0.0753806
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.67 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 21

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.65 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755112, upper bound: 0.0754053
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755112, upper bound: 0.0754053
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.68 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 21

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.65 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0756055, upper bound: 0.0753806
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0756055, upper bound: 0.0753806
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.69 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 21

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.65 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755838, upper bound: 0.0754446
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755838, upper bound: 0.0754446
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.68 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 21

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.66 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753041, upper bound: 0.0755510
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753041, upper bound: 0.0755510
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.70 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 21

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.66 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753148, upper bound: 0.0755749
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753148, upper bound: 0.0755749
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.70 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 21

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.66 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753067, upper bound: 0.0755199
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753067, upper bound: 0.0755199
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.70 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 21

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.66 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753148, upper bound: 0.0755387
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753148, upper bound: 0.0755387
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.70 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 21

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.66 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755079, upper bound: 0.0753218
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755079, upper bound: 0.0753218
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.72 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 21

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.66 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755585, upper bound: 0.0753426
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755585, upper bound: 0.0753426
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.71 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 21

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.67 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0756222, upper bound: 0.0753218
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0756222, upper bound: 0.0753218
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.72 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 21

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.67 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0756152, upper bound: 0.0753782
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0756152, upper bound: 0.0753782
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.72 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 21

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.67 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753782, upper bound: 0.0756152
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753782, upper bound: 0.0756152
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.72 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 21

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.67 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753218, upper bound: 0.0756222
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753218, upper bound: 0.0756222
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.73 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 21

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.67 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753426, upper bound: 0.0755585
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753426, upper bound: 0.0755585
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.74 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 21

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.67 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753218, upper bound: 0.0755079
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753218, upper bound: 0.0755079
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.73 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 21

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.67 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755387, upper bound: 0.0753148
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755387, upper bound: 0.0753148
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.74 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 21

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.68 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755199, upper bound: 0.0753067
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755199, upper bound: 0.0753067
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.75 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 21

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.68 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755749, upper bound: 0.0753148
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755749, upper bound: 0.0753148
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.76 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 21

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.68 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755510, upper bound: 0.0753041
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755510, upper bound: 0.0753041
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.76 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 21

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.68 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0754446, upper bound: 0.0755838
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0754446, upper bound: 0.0755838
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.75 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 21

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.69 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753806, upper bound: 0.0756055
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753806, upper bound: 0.0756055
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 21

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.68 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0754053, upper bound: 0.0755112
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0754053, upper bound: 0.0755112
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 21

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.69 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753806, upper bound: 0.0755079
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753806, upper bound: 0.0755079
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 21

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.69 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0756007, upper bound: 0.0752888
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0756007, upper bound: 0.0752888
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 21

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.69 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755844, upper bound: 0.0752894
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755844, upper bound: 0.0752894
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 21

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.69 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0756241, upper bound: 0.0752888
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0756241, upper bound: 0.0752888
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.79 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 21

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.69 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0756043, upper bound: 0.0752894
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0756043, upper bound: 0.0752894
time: 0.24 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 3.36 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0752894, upper bound: 0.0756043
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0752894, upper bound: 0.0756043
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0752888, upper bound: 0.0756241
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0752888, upper bound: 0.0756241
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0752894, upper bound: 0.0755844
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0752894, upper bound: 0.0755844
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0752888, upper bound: 0.0756007
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0752888, upper bound: 0.0756007
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0755079, upper bound: 0.0753806
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0755079, upper bound: 0.0753806
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0755112, upper bound: 0.0754053
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0755112, upper bound: 0.0754053
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0756055, upper bound: 0.0753806
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0756055, upper bound: 0.0753806
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0755838, upper bound: 0.0754446
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0755838, upper bound: 0.0754446
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0753041, upper bound: 0.0755510
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0753041, upper bound: 0.0755510
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0753148, upper bound: 0.0755749
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0753148, upper bound: 0.0755749
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0753067, upper bound: 0.0755199
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0753067, upper bound: 0.0755199
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0753148, upper bound: 0.0755387
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0753148, upper bound: 0.0755387
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0755079, upper bound: 0.0753218
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0755079, upper bound: 0.0753218
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0755585, upper bound: 0.0753426
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0755585, upper bound: 0.0753426
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0756222, upper bound: 0.0753218
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0756222, upper bound: 0.0753218
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0756152, upper bound: 0.0753782
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0756152, upper bound: 0.0753782
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0753782, upper bound: 0.0756152
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0753782, upper bound: 0.0756152
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0753218, upper bound: 0.0756222
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0753218, upper bound: 0.0756222
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0753426, upper bound: 0.0755585
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0753426, upper bound: 0.0755585
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0753218, upper bound: 0.0755079
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0753218, upper bound: 0.0755079
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0755387, upper bound: 0.0753148
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0755387, upper bound: 0.0753148
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0755199, upper bound: 0.0753067
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0755199, upper bound: 0.0753067
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0755749, upper bound: 0.0753148
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0755749, upper bound: 0.0753148
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0755510, upper bound: 0.0753041
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0755510, upper bound: 0.0753041
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0754446, upper bound: 0.0755838
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0754446, upper bound: 0.0755838
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0753806, upper bound: 0.0756055
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0753806, upper bound: 0.0756055
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0754053, upper bound: 0.0755112
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0754053, upper bound: 0.0755112
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0753806, upper bound: 0.0755079
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0753806, upper bound: 0.0755079
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0756007, upper bound: 0.0752888
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0756007, upper bound: 0.0752888
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0755844, upper bound: 0.0752894
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0755844, upper bound: 0.0752894
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0756241, upper bound: 0.0752888
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0756241, upper bound: 0.0752888
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0756043, upper bound: 0.0752894
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -0.0756043, upper bound: 0.0752894

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.75 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0751760, upper bound: 0.0756043
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0752232, upper bound: 0.0756043
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.73 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0751760, upper bound: 0.0756043
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0752232, upper bound: 0.0756043
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.73 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0752114, upper bound: 0.0756241
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0752261, upper bound: 0.0756079
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.73 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0752114, upper bound: 0.0756241
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0752261, upper bound: 0.0756079
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.73 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0751759, upper bound: 0.0755843
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0752158, upper bound: 0.0755844
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.74 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0751759, upper bound: 0.0755843
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0752158, upper bound: 0.0755844
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.74 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0752061, upper bound: 0.0756007
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0752261, upper bound: 0.0755884
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.75 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0752061, upper bound: 0.0756007
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0752261, upper bound: 0.0755884
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.76 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0754684, upper bound: 0.0753402
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755079, upper bound: 0.0753198
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.76 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0754684, upper bound: 0.0753402
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755079, upper bound: 0.0753198
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0754961, upper bound: 0.0753408
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755112, upper bound: 0.0753336
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0754961, upper bound: 0.0753408
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755112, upper bound: 0.0753336
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755776, upper bound: 0.0753443
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755966, upper bound: 0.0753415
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755776, upper bound: 0.0753443
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755966, upper bound: 0.0753415
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.78 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755769, upper bound: 0.0753703
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755825, upper bound: 0.0753995
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.78 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755769, upper bound: 0.0753703
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755825, upper bound: 0.0753995
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.78 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0751839, upper bound: 0.0755510
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0752312, upper bound: 0.0755436
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.79 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0751839, upper bound: 0.0755510
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0752312, upper bound: 0.0755436
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.79 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0752378, upper bound: 0.0755749
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0752470, upper bound: 0.0755505
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.80 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0752378, upper bound: 0.0755749
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0752470, upper bound: 0.0755505
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.80 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0751889, upper bound: 0.0755199
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0752227, upper bound: 0.0755057
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.82 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0751889, upper bound: 0.0755199
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0752227, upper bound: 0.0755057
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.81 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0752357, upper bound: 0.0755387
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0752470, upper bound: 0.0755060
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.81 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0752357, upper bound: 0.0755387
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0752470, upper bound: 0.0755060
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.83 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0754741, upper bound: 0.0752722
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755079, upper bound: 0.0752390
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.82 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0754741, upper bound: 0.0752722
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755079, upper bound: 0.0752390
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.85 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755434, upper bound: 0.0752814
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755571, upper bound: 0.0752612
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.83 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755434, upper bound: 0.0752814
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755571, upper bound: 0.0752612
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.84 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0756075, upper bound: 0.0752722
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0756222, upper bound: 0.0752690
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.84 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0756075, upper bound: 0.0752722
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0756222, upper bound: 0.0752690
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.85 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0756047, upper bound: 0.0753105
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0756152, upper bound: 0.0753310
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.86 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0756047, upper bound: 0.0753105
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0756152, upper bound: 0.0753310
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.85 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753310, upper bound: 0.0756152
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753105, upper bound: 0.0756047
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.85 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753310, upper bound: 0.0756152
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753105, upper bound: 0.0756047
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.87 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0752690, upper bound: 0.0756222
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0752722, upper bound: 0.0756075
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.87 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0752690, upper bound: 0.0756222
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0752722, upper bound: 0.0756075
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.86 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0752612, upper bound: 0.0755571
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0752814, upper bound: 0.0755434
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.87 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0752612, upper bound: 0.0755571
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0752814, upper bound: 0.0755434
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.87 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0752390, upper bound: 0.0755079
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0752722, upper bound: 0.0754741
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.87 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0752390, upper bound: 0.0755079
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0752722, upper bound: 0.0754741
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755060, upper bound: 0.0752470
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755387, upper bound: 0.0752357
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.88 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755060, upper bound: 0.0752470
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755387, upper bound: 0.0752357
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755057, upper bound: 0.0752227
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755199, upper bound: 0.0751889
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755057, upper bound: 0.0752227
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755199, upper bound: 0.0751889
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.90 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755505, upper bound: 0.0752470
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755749, upper bound: 0.0752378
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.90 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755505, upper bound: 0.0752470
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755749, upper bound: 0.0752378
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.91 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755436, upper bound: 0.0752312
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755510, upper bound: 0.0751839
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.91 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755436, upper bound: 0.0752312
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755510, upper bound: 0.0751839
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.92 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753995, upper bound: 0.0755825
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753703, upper bound: 0.0755769
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.93 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753995, upper bound: 0.0755825
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753703, upper bound: 0.0755769
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.93 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753415, upper bound: 0.0755966
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753443, upper bound: 0.0755776
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.92 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753415, upper bound: 0.0755966
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753443, upper bound: 0.0755776
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.93 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753336, upper bound: 0.0755112
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753408, upper bound: 0.0754961
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.94 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753336, upper bound: 0.0755112
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753408, upper bound: 0.0754961
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.93 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753198, upper bound: 0.0755079
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753402, upper bound: 0.0754684
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.94 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753198, upper bound: 0.0755079
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753402, upper bound: 0.0754684
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.95 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755884, upper bound: 0.0752261
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0756007, upper bound: 0.0752061
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.94 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755884, upper bound: 0.0752261
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0756007, upper bound: 0.0752061
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.96 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755844, upper bound: 0.0752158
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755843, upper bound: 0.0751759
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.95 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755844, upper bound: 0.0752158
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0755843, upper bound: 0.0751759
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.96 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0756079, upper bound: 0.0752261
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0756241, upper bound: 0.0752114
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.97 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0756079, upper bound: 0.0752261
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0756241, upper bound: 0.0752114
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.97 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0756043, upper bound: 0.0752232
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0756043, upper bound: 0.0751760
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.96 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0756043, upper bound: 0.0752232
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0756043, upper bound: 0.0751760
time: 0.26 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 2.65 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0751760, upper bound: 0.0756043
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0752232, upper bound: 0.0756043
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0751760, upper bound: 0.0756043
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0752232, upper bound: 0.0756043
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0752114, upper bound: 0.0756241
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0752261, upper bound: 0.0756079
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0752114, upper bound: 0.0756241
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0752261, upper bound: 0.0756079
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0751759, upper bound: 0.0755843
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0752158, upper bound: 0.0755844
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0751759, upper bound: 0.0755843
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0752158, upper bound: 0.0755844
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0752061, upper bound: 0.0756007
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0752261, upper bound: 0.0755884
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0752061, upper bound: 0.0756007
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0752261, upper bound: 0.0755884
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0754684, upper bound: 0.0753402
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0755079, upper bound: 0.0753198
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0754684, upper bound: 0.0753402
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0755079, upper bound: 0.0753198
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0754961, upper bound: 0.0753408
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0755112, upper bound: 0.0753336
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0754961, upper bound: 0.0753408
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0755112, upper bound: 0.0753336
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0755776, upper bound: 0.0753443
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0755966, upper bound: 0.0753415
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0755776, upper bound: 0.0753443
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0755966, upper bound: 0.0753415
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0755769, upper bound: 0.0753703
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0755825, upper bound: 0.0753995
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0755769, upper bound: 0.0753703
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0755825, upper bound: 0.0753995
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0751839, upper bound: 0.0755510
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0752312, upper bound: 0.0755436
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0751839, upper bound: 0.0755510
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0752312, upper bound: 0.0755436
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0752378, upper bound: 0.0755749
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0752470, upper bound: 0.0755505
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0752378, upper bound: 0.0755749
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0752470, upper bound: 0.0755505
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0751889, upper bound: 0.0755199
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0752227, upper bound: 0.0755057
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0751889, upper bound: 0.0755199
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0752227, upper bound: 0.0755057
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0752357, upper bound: 0.0755387
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0752470, upper bound: 0.0755060
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0752357, upper bound: 0.0755387
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0752470, upper bound: 0.0755060
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0754741, upper bound: 0.0752722
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0755079, upper bound: 0.0752390
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0754741, upper bound: 0.0752722
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0755079, upper bound: 0.0752390
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0755434, upper bound: 0.0752814
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0755571, upper bound: 0.0752612
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0755434, upper bound: 0.0752814
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0755571, upper bound: 0.0752612
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0756075, upper bound: 0.0752722
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0756222, upper bound: 0.0752690
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0756075, upper bound: 0.0752722
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0756222, upper bound: 0.0752690
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0756047, upper bound: 0.0753105
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0756152, upper bound: 0.0753310
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0756047, upper bound: 0.0753105
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0756152, upper bound: 0.0753310
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0753310, upper bound: 0.0756152
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0753105, upper bound: 0.0756047
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0753310, upper bound: 0.0756152
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0753105, upper bound: 0.0756047
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0752690, upper bound: 0.0756222
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0752722, upper bound: 0.0756075
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0752690, upper bound: 0.0756222
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0752722, upper bound: 0.0756075
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0752612, upper bound: 0.0755571
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0752814, upper bound: 0.0755434
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0752612, upper bound: 0.0755571
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0752814, upper bound: 0.0755434
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0752390, upper bound: 0.0755079
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0752722, upper bound: 0.0754741
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0752390, upper bound: 0.0755079
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0752722, upper bound: 0.0754741
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0755060, upper bound: 0.0752470
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0755387, upper bound: 0.0752357
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0755060, upper bound: 0.0752470
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0755387, upper bound: 0.0752357
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0755057, upper bound: 0.0752227
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0755199, upper bound: 0.0751889
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0755057, upper bound: 0.0752227
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0755199, upper bound: 0.0751889
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0755505, upper bound: 0.0752470
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0755749, upper bound: 0.0752378
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0755505, upper bound: 0.0752470
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0755749, upper bound: 0.0752378
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0755436, upper bound: 0.0752312
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0755510, upper bound: 0.0751839
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0755436, upper bound: 0.0752312
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0755510, upper bound: 0.0751839
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0753995, upper bound: 0.0755825
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0753703, upper bound: 0.0755769
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0753995, upper bound: 0.0755825
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0753703, upper bound: 0.0755769
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0753415, upper bound: 0.0755966
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0753443, upper bound: 0.0755776
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0753415, upper bound: 0.0755966
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0753443, upper bound: 0.0755776
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0753336, upper bound: 0.0755112
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0753408, upper bound: 0.0754961
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0753336, upper bound: 0.0755112
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0753408, upper bound: 0.0754961
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0753198, upper bound: 0.0755079
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0753402, upper bound: 0.0754684
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0753198, upper bound: 0.0755079
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0753402, upper bound: 0.0754684
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0755884, upper bound: 0.0752261
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0756007, upper bound: 0.0752061
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0755884, upper bound: 0.0752261
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0756007, upper bound: 0.0752061
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0755844, upper bound: 0.0752158
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0755843, upper bound: 0.0751759
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0755844, upper bound: 0.0752158
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0755843, upper bound: 0.0751759
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0756079, upper bound: 0.0752261
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0756241, upper bound: 0.0752114
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0756079, upper bound: 0.0752261
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0756241, upper bound: 0.0752114
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0756043, upper bound: 0.0752232
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0756043, upper bound: 0.0751760
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0756043, upper bound: 0.0752232
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -0.0756043, upper bound: 0.0751760

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.88 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 31

### Candidate
type: DSZ, layer: 5, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0737851, upper bound: 0.0742421
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0737534, upper bound: 0.0741854
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.84 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 31

### Candidate
type: DSZ, layer: 5, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0737786, upper bound: 0.0742210
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0737677, upper bound: 0.0741848
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.85 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 31

### Candidate
type: DSZ, layer: 5, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0737877, upper bound: 0.0742421
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0737796, upper bound: 0.0741854
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.86 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 31

### Candidate
type: DSZ, layer: 5, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0737786, upper bound: 0.0742181
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0737770, upper bound: 0.0741848
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.86 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 31

### Candidate
type: DSZ, layer: 5, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0737850, upper bound: 0.0746001
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0737609, upper bound: 0.0744057
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.86 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 31

### Candidate
type: DSZ, layer: 5, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0737781, upper bound: 0.0746684
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0737704, upper bound: 0.0745588
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.88 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 31

### Candidate
type: DSZ, layer: 5, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0737876, upper bound: 0.0746001
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0737796, upper bound: 0.0744057
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.87 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 31

### Candidate
type: DSZ, layer: 5, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0737781, upper bound: 0.0746684
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0737772, upper bound: 0.0745588
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.88 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 31

### Candidate
type: DSZ, layer: 5, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0737915, upper bound: 0.0740697
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0737534, upper bound: 0.0740970
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 31

### Candidate
type: DSZ, layer: 5, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0737814, upper bound: 0.0740572
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0737534, upper bound: 0.0740958
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 31

### Candidate
type: DSZ, layer: 5, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0737921, upper bound: 0.0740697
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0737802, upper bound: 0.0740970
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 31

### Candidate
type: DSZ, layer: 5, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0737814, upper bound: 0.0740460
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0737673, upper bound: 0.0740951
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.90 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 31

### Candidate
type: DSZ, layer: 5, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0737926, upper bound: 0.0740686
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0737738, upper bound: 0.0741610
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.91 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 31

### Candidate
type: DSZ, layer: 5, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0737809, upper bound: 0.0740492
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0737764, upper bound: 0.0741847
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.91 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 31

### Candidate
type: DSZ, layer: 5, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0737932, upper bound: 0.0740686
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0737861, upper bound: 0.0741610
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.91 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 31

### Candidate
type: DSZ, layer: 5, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0737809, upper bound: 0.0740377
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0737783, upper bound: 0.0741847
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.91 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 31

### Candidate
type: DSZ, layer: 5, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0740488, upper bound: 0.0738161
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0740374, upper bound: 0.0738526
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.92 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 31

### Candidate
type: DSZ, layer: 5, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0740573, upper bound: 0.0738188
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0740580, upper bound: 0.0738573
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.92 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 31

### Candidate
type: DSZ, layer: 5, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0740533, upper bound: 0.0738146
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0740470, upper bound: 0.0738526
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.92 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 31

### Candidate
type: DSZ, layer: 5, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0740594, upper bound: 0.0738088
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0740594, upper bound: 0.0738569
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.93 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 31

### Candidate
type: DSZ, layer: 5, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0740611, upper bound: 0.0738186
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0740514, upper bound: 0.0740445
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.94 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 31

### Candidate
type: DSZ, layer: 5, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0740661, upper bound: 0.0738194
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0740661, upper bound: 0.0742716
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.94 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 31

### Candidate
type: DSZ, layer: 5, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0740667, upper bound: 0.0738180
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0740643, upper bound: 0.0740445
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.94 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 31

### Candidate
type: DSZ, layer: 5, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0740696, upper bound: 0.0738123
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0740694, upper bound: 0.0742716
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.95 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 31

### Candidate
type: DSZ, layer: 5, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0743082, upper bound: 0.0737948
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0746124, upper bound: 0.0739167
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.95 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 31

### Candidate
type: DSZ, layer: 5, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0741722, upper bound: 0.0737949
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0745799, upper bound: 0.0739315
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.95 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 31

### Candidate
type: DSZ, layer: 5, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0743082, upper bound: 0.0737923
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0746127, upper bound: 0.0739167
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.96 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 31

### Candidate
type: DSZ, layer: 5, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0741722, upper bound: 0.0737729
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0745799, upper bound: 0.0739306
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 1.96 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 25
type: DSZ, layer: 5, pos: 31
type: DSZ, layer: 5, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 25

### Candidate
type: DSZ, layer: 5, pos: 31

### Candidate
type: DSZ, layer: 5, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0741995, upper bound: 0.0737990
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0743807, upper bound: 0.0740443
time: 0.27 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 2.66 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0737851, upper bound: 0.0742421
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0737534, upper bound: 0.0741854
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0737786, upper bound: 0.0742210
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0737677, upper bound: 0.0741848
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0737877, upper bound: 0.0742421
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0737796, upper bound: 0.0741854
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0737786, upper bound: 0.0742181
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0737770, upper bound: 0.0741848
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0737850, upper bound: 0.0746001
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0737609, upper bound: 0.0744057
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0737781, upper bound: 0.0746684
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0737704, upper bound: 0.0745588
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0737876, upper bound: 0.0746001
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0737796, upper bound: 0.0744057
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0737781, upper bound: 0.0746684
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0737772, upper bound: 0.0745588
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0737915, upper bound: 0.0740697
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0737534, upper bound: 0.0740970
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0737814, upper bound: 0.0740572
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0737534, upper bound: 0.0740958
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0737921, upper bound: 0.0740697
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0737802, upper bound: 0.0740970
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0737814, upper bound: 0.0740460
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0737673, upper bound: 0.0740951
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0737926, upper bound: 0.0740686
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0737738, upper bound: 0.0741610
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0737809, upper bound: 0.0740492
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0737764, upper bound: 0.0741847
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0737932, upper bound: 0.0740686
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0737861, upper bound: 0.0741610
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0737809, upper bound: 0.0740377
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0737783, upper bound: 0.0741847
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0740488, upper bound: 0.0738161
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0740374, upper bound: 0.0738526
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0740573, upper bound: 0.0738188
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0740580, upper bound: 0.0738573
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0740533, upper bound: 0.0738146
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0740470, upper bound: 0.0738526
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0740594, upper bound: 0.0738088
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0740594, upper bound: 0.0738569
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0740611, upper bound: 0.0738186
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0740514, upper bound: 0.0740445
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0740661, upper bound: 0.0738194
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0740661, upper bound: 0.0742716
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0740667, upper bound: 0.0738180
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0740643, upper bound: 0.0740445
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0740696, upper bound: 0.0738123
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0740694, upper bound: 0.0742716
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0743082, upper bound: 0.0737948
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0746124, upper bound: 0.0739167
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0741722, upper bound: 0.0737949
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0745799, upper bound: 0.0739315
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0743082, upper bound: 0.0737923
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0746127, upper bound: 0.0739167
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0741722, upper bound: 0.0737729
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0745799, upper bound: 0.0739306
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0741995, upper bound: 0.0737990
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0743807, upper bound: 0.0740443
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0755825, upper bound: 0.0753995
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0755769, upper bound: 0.0753703
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0755825, upper bound: 0.0753995
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0751839, upper bound: 0.0755510
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0752312, upper bound: 0.0755436
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0751839, upper bound: 0.0755510
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0752312, upper bound: 0.0755436
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0752378, upper bound: 0.0755749
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0752470, upper bound: 0.0755505
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0752378, upper bound: 0.0755749
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0752470, upper bound: 0.0755505
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0751889, upper bound: 0.0755199
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0752227, upper bound: 0.0755057
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0751889, upper bound: 0.0755199
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0752227, upper bound: 0.0755057
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0752357, upper bound: 0.0755387
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0752470, upper bound: 0.0755060
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0752357, upper bound: 0.0755387
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0752470, upper bound: 0.0755060
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0754741, upper bound: 0.0752722
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0755079, upper bound: 0.0752390
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0754741, upper bound: 0.0752722
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0755079, upper bound: 0.0752390
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0755434, upper bound: 0.0752814
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0755571, upper bound: 0.0752612
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0755434, upper bound: 0.0752814
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0755571, upper bound: 0.0752612
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0756075, upper bound: 0.0752722
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0756222, upper bound: 0.0752690
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0756075, upper bound: 0.0752722
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0756222, upper bound: 0.0752690
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0756047, upper bound: 0.0753105
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0756152, upper bound: 0.0753310
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0756047, upper bound: 0.0753105
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0756152, upper bound: 0.0753310
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0753310, upper bound: 0.0756152
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0753105, upper bound: 0.0756047
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0753310, upper bound: 0.0756152
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0753105, upper bound: 0.0756047
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0752690, upper bound: 0.0756222
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0752722, upper bound: 0.0756075
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0752690, upper bound: 0.0756222
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0752722, upper bound: 0.0756075
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0752612, upper bound: 0.0755571
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0752814, upper bound: 0.0755434
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0752612, upper bound: 0.0755571
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0752814, upper bound: 0.0755434
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0752390, upper bound: 0.0755079
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0752722, upper bound: 0.0754741
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0752390, upper bound: 0.0755079
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0752722, upper bound: 0.0754741
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0755060, upper bound: 0.0752470
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0755387, upper bound: 0.0752357
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0755060, upper bound: 0.0752470
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0755387, upper bound: 0.0752357
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0755057, upper bound: 0.0752227
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0755199, upper bound: 0.0751889
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0755057, upper bound: 0.0752227
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0755199, upper bound: 0.0751889
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0755505, upper bound: 0.0752470
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0755749, upper bound: 0.0752378
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0755505, upper bound: 0.0752470
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0755749, upper bound: 0.0752378
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0755436, upper bound: 0.0752312
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0755510, upper bound: 0.0751839
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0755436, upper bound: 0.0752312
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0755510, upper bound: 0.0751839
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0753995, upper bound: 0.0755825
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0753703, upper bound: 0.0755769
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0753995, upper bound: 0.0755825
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0753703, upper bound: 0.0755769
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0753415, upper bound: 0.0755966
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0753443, upper bound: 0.0755776
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0753415, upper bound: 0.0755966
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0753443, upper bound: 0.0755776
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0753336, upper bound: 0.0755112
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0753408, upper bound: 0.0754961
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0753336, upper bound: 0.0755112
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0753408, upper bound: 0.0754961
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0753198, upper bound: 0.0755079
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0753402, upper bound: 0.0754684
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0753198, upper bound: 0.0755079
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0753402, upper bound: 0.0754684
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0755884, upper bound: 0.0752261
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0756007, upper bound: 0.0752061
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0755884, upper bound: 0.0752261
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0756007, upper bound: 0.0752061
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0755844, upper bound: 0.0752158
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0755843, upper bound: 0.0751759
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0755844, upper bound: 0.0752158
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0755843, upper bound: 0.0751759
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0756079, upper bound: 0.0752261
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0756241, upper bound: 0.0752114
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0756079, upper bound: 0.0752261
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0756241, upper bound: 0.0752114
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0756043, upper bound: 0.0752232
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0756043, upper bound: 0.0751760
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0756043, upper bound: 0.0752232
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 0, lower bound: -0.0756043, upper bound: 0.0751760

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 2.42 + 417.58 = 420.00 seconds
