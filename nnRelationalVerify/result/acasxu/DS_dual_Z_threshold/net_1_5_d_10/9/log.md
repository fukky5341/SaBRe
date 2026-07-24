## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_5.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 9)
Time budget: 420 seconds
Split limit: 100
Threshold: 27.7691976323


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604)
1: (-11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856)
2: (-9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369)
3: (-10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898)
4: (-8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.68 + 1.58 = 2.26 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -27.8527559, upper bound: 27.8527559

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8198125, upper bound: 27.8198125
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8198125, upper bound: 27.8198125
time: 0.75 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.68 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.68
Output dim: 0, lower bound: -27.8198125, upper bound: 27.8198125
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.68
Output dim: 0, lower bound: -27.8198125, upper bound: 27.8198125

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8164844, upper bound: 27.8164844
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8164844, upper bound: 27.8164844
time: 0.80 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8164844, upper bound: 27.8164844
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8164844, upper bound: 27.8164844
time: 0.80 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.14 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.14
Output dim: 0, lower bound: -27.8164844, upper bound: 27.8164844
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.14
Output dim: 0, lower bound: -27.8164844, upper bound: 27.8164844
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.14
Output dim: 0, lower bound: -27.8164844, upper bound: 27.8164844
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.14
Output dim: 0, lower bound: -27.8164844, upper bound: 27.8164844

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.87 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.47 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.55 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.55
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.55
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.55
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.55
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.55
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.55
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.55
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.55
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.93 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.89 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.35 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 1.01 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 1.03 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.64 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 2.02 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.89 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.44 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 1.71 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 2.26 + 126.90 = 129.16 seconds
