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
execution time: IAR + RelationalAnalysis = 1.00 + 1.61 = 2.61 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -27.8527559, upper bound: 27.8527559

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8521223, upper bound: 27.8521223
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8521223, upper bound: 27.8527559
time: 0.51 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.24 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.24
Output dim: 0, lower bound: -27.8521223, upper bound: 27.8521223
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.24
Output dim: 0, lower bound: -27.8521223, upper bound: 27.8527559

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8047160, upper bound: 27.8047160
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8047160, upper bound: 27.8047160
time: 0.63 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8166164, upper bound: 27.8166164
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8166164, upper bound: 27.8166164
time: 0.75 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.36 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.36
Output dim: 0, lower bound: -27.8047160, upper bound: 27.8047160
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.36
Output dim: 0, lower bound: -27.8047160, upper bound: 27.8047160
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.36
Output dim: 0, lower bound: -27.8166164, upper bound: 27.8166164
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.36
Output dim: 0, lower bound: -27.8166164, upper bound: 27.8166164

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8047160, upper bound: 27.8047160
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8047160, upper bound: 27.8047160
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8047160, upper bound: 27.8047160
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8047160, upper bound: 27.8047160
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8105578, upper bound: 27.8105578
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8105578, upper bound: 27.8105578
time: 0.98 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8105578, upper bound: 27.8105578
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8105578, upper bound: 27.8105578
time: 0.62 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.11 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.11
Output dim: 0, lower bound: -27.8047160, upper bound: 27.8047160
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.11
Output dim: 0, lower bound: -27.8047160, upper bound: 27.8047160
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.11
Output dim: 0, lower bound: -27.8047160, upper bound: 27.8047160
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.11
Output dim: 0, lower bound: -27.8047160, upper bound: 27.8047160
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.11
Output dim: 0, lower bound: -27.8105578, upper bound: 27.8105578
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.11
Output dim: 0, lower bound: -27.8105578, upper bound: 27.8105578
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.11
Output dim: 0, lower bound: -27.8105578, upper bound: 27.8105578
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.11
Output dim: 0, lower bound: -27.8105578, upper bound: 27.8105578

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8043713, upper bound: 27.8043713
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8043713, upper bound: 27.8043713
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.3120452, upper bound: 27.3120452
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.3120452, upper bound: 27.3120452
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.6494830, upper bound: 27.6494830
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.6494830, upper bound: 27.6494830
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.3120452, upper bound: 27.3120452
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.3120452, upper bound: 27.3120452
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8105036, upper bound: 27.8105036
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8105036, upper bound: 27.8105036
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8105578, upper bound: 27.8105578
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8105578, upper bound: 27.8105578
time: 0.46 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.05 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -27.8043713, upper bound: 27.8043713
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -27.8043713, upper bound: 27.8043713
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.05
Output dim: 0, lower bound: -27.3120452, upper bound: 27.3120452
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 2.05
Output dim: 0, lower bound: -27.3120452, upper bound: 27.3120452
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.05
Output dim: 0, lower bound: -27.6494830, upper bound: 27.6494830
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 2.05
Output dim: 0, lower bound: -27.6494830, upper bound: 27.6494830
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.05
Output dim: 0, lower bound: -27.3120452, upper bound: 27.3120452
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 2.05
Output dim: 0, lower bound: -27.3120452, upper bound: 27.3120452
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -27.8105036, upper bound: 27.8105036
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -27.8105036, upper bound: 27.8105036
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -27.8105578, upper bound: 27.8105578
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -27.8105578, upper bound: 27.8105578

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -26.8495515, upper bound: 26.8495515
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -26.8495515, upper bound: 26.8495515
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.1913796, upper bound: 27.1913796
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.1913796, upper bound: 27.1913796
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8105036, upper bound: 27.8105036
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8105036, upper bound: 27.8105036
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8105036, upper bound: 27.8105036
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8105036, upper bound: 27.8105036
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8105036, upper bound: 27.8105036
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8105036, upper bound: 27.8105036
time: 0.75 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 2.37 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.37
Output dim: 0, lower bound: -26.8495515, upper bound: 26.8495515
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.37
Output dim: 0, lower bound: -26.8495515, upper bound: 26.8495515
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.37
Output dim: 0, lower bound: -27.1913796, upper bound: 27.1913796
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.37
Output dim: 0, lower bound: -27.1913796, upper bound: 27.1913796
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -27.8105036, upper bound: 27.8105036
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -27.8105036, upper bound: 27.8105036
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -27.8105036, upper bound: 27.8105036
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -27.8105036, upper bound: 27.8105036
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -27.8105036, upper bound: 27.8105036
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -27.8105036, upper bound: 27.8105036

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8105036, upper bound: 27.8105036
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8105036, upper bound: 27.8105036
time: 1.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8105036, upper bound: 27.8105036
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8105036, upper bound: 27.8105036
time: 1.13 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8105036, upper bound: 27.8105036
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8105036, upper bound: 27.8105036
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8105036, upper bound: 27.8105036
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8105036, upper bound: 27.8105036
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 1.24 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 3.75 seconds
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.75
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.75
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.75
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.75
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.75
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.75
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.75
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.75
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.75
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.75
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.75
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.75
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.75
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.75
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.75
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.75
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.75
Output dim: 0, lower bound: -27.8105036, upper bound: 27.8105036
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.75
Output dim: 0, lower bound: -27.8105036, upper bound: 27.8105036
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.75
Output dim: 0, lower bound: -27.8105036, upper bound: 27.8105036
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.75
Output dim: 0, lower bound: -27.8105036, upper bound: 27.8105036
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.75
Output dim: 0, lower bound: -27.8105036, upper bound: 27.8105036
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.75
Output dim: 0, lower bound: -27.8105036, upper bound: 27.8105036
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.75
Output dim: 0, lower bound: -27.8105036, upper bound: 27.8105036
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.75
Output dim: 0, lower bound: -27.8105036, upper bound: 27.8105036
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.75
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.75
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.75
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.75
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.75
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.75
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.75
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.75
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.98 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8105036, upper bound: 27.8105036
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8105036, upper bound: 27.8105036
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8105036, upper bound: 27.8105036
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8105036, upper bound: 27.8105036
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8105036, upper bound: 27.8105036
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8105036, upper bound: 27.8105036
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8105036, upper bound: 27.8105036
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8105036, upper bound: 27.8105036
time: 0.70 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
time: 0.92 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
time: 0.92 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
time: 0.58 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 2.49 seconds
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.49
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.49
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.49
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.49
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.49
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.49
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.49
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.49
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.49
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.49
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.49
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.49
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.49
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.49
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.49
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.49
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.49
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.49
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.49
Output dim: 0, lower bound: -27.8105036, upper bound: 27.8105036
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.49
Output dim: 0, lower bound: -27.8105036, upper bound: 27.8105036
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.49
Output dim: 0, lower bound: -27.8105036, upper bound: 27.8105036
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.49
Output dim: 0, lower bound: -27.8105036, upper bound: 27.8105036
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.49
Output dim: 0, lower bound: -27.8105036, upper bound: 27.8105036
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.49
Output dim: 0, lower bound: -27.8105036, upper bound: 27.8105036
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.49
Output dim: 0, lower bound: -27.8105036, upper bound: 27.8105036
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.49
Output dim: 0, lower bound: -27.8105036, upper bound: 27.8105036
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.49
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.49
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.49
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.49
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.49
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.49
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.49
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.49
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.49
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.49
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.49
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.49
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.49
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.49
Output dim: 0, lower bound: -27.7804949, upper bound: 27.7804949

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.94 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 1.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604
1: -11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856
2: -9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369
3: -10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898
4: -8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
time: 0.46 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 2.76 seconds
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -27.7635121, upper bound: 27.7635121

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 2.61 + 195.86 = 198.47 seconds
