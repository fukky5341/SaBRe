## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_8.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 3)
Time budget: 420 seconds
Split limit: 100
Threshold: 0.00109514


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557)
1: (-0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170)
2: (-0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386)
3: (-0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066)
4: (-0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.96 + 0.61 = 1.57 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0012884, upper bound: 0.0012884

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012868, upper bound: 0.0012833
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012833, upper bound: 0.0012868
time: 0.20 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.48 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.48
Output dim: 0, lower bound: -0.0012868, upper bound: 0.0012833
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.48
Output dim: 0, lower bound: -0.0012833, upper bound: 0.0012868

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012820, upper bound: 0.0012783
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012760, upper bound: 0.0012819
time: 0.18 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012819, upper bound: 0.0012760
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012783, upper bound: 0.0012820
time: 0.16 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.32 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.32
Output dim: 0, lower bound: -0.0012820, upper bound: 0.0012783
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.32
Output dim: 0, lower bound: -0.0012760, upper bound: 0.0012819
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.32
Output dim: 0, lower bound: -0.0012819, upper bound: 0.0012760
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.32
Output dim: 0, lower bound: -0.0012783, upper bound: 0.0012820

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012805, upper bound: 0.0012754
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012777, upper bound: 0.0012761
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012745, upper bound: 0.0012772
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012739, upper bound: 0.0012801
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012801, upper bound: 0.0012739
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012772, upper bound: 0.0012745
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012761, upper bound: 0.0012777
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012754, upper bound: 0.0012805
time: 0.17 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 1.81 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.81
Output dim: 0, lower bound: -0.0012805, upper bound: 0.0012754
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.81
Output dim: 0, lower bound: -0.0012777, upper bound: 0.0012761
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.81
Output dim: 0, lower bound: -0.0012745, upper bound: 0.0012772
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.81
Output dim: 0, lower bound: -0.0012739, upper bound: 0.0012801
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.81
Output dim: 0, lower bound: -0.0012801, upper bound: 0.0012739
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.81
Output dim: 0, lower bound: -0.0012772, upper bound: 0.0012745
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.81
Output dim: 0, lower bound: -0.0012761, upper bound: 0.0012777
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.81
Output dim: 0, lower bound: -0.0012754, upper bound: 0.0012805

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Candidate
type: DSZ, layer: 1, pos: 31

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.37 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012590, upper bound: 0.0012567
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012591, upper bound: 0.0012566
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Candidate
type: DSZ, layer: 1, pos: 31

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.41 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012565, upper bound: 0.0012571
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012574, upper bound: 0.0012571
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Candidate
type: DSZ, layer: 1, pos: 31

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.37 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012564, upper bound: 0.0012569
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012564, upper bound: 0.0012566
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Candidate
type: DSZ, layer: 1, pos: 31

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.38 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012563, upper bound: 0.0012582
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012564, upper bound: 0.0012582
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Candidate
type: DSZ, layer: 1, pos: 31

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.38 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012582, upper bound: 0.0012564
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012582, upper bound: 0.0012563
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Candidate
type: DSZ, layer: 1, pos: 31

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.38 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012566, upper bound: 0.0012564
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012569, upper bound: 0.0012564
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Candidate
type: DSZ, layer: 1, pos: 31

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.38 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012571, upper bound: 0.0012574
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012571, upper bound: 0.0012565
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Candidate
type: DSZ, layer: 1, pos: 31

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.38 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012566, upper bound: 0.0012591
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012567, upper bound: 0.0012590
time: 0.18 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 1.69 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.69
Output dim: 0, lower bound: -0.0012590, upper bound: 0.0012567
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.69
Output dim: 0, lower bound: -0.0012591, upper bound: 0.0012566
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.69
Output dim: 0, lower bound: -0.0012565, upper bound: 0.0012571
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.69
Output dim: 0, lower bound: -0.0012574, upper bound: 0.0012571
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.69
Output dim: 0, lower bound: -0.0012564, upper bound: 0.0012569
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.69
Output dim: 0, lower bound: -0.0012564, upper bound: 0.0012566
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.69
Output dim: 0, lower bound: -0.0012563, upper bound: 0.0012582
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.69
Output dim: 0, lower bound: -0.0012564, upper bound: 0.0012582
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.69
Output dim: 0, lower bound: -0.0012582, upper bound: 0.0012564
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.69
Output dim: 0, lower bound: -0.0012582, upper bound: 0.0012563
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.69
Output dim: 0, lower bound: -0.0012566, upper bound: 0.0012564
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.69
Output dim: 0, lower bound: -0.0012569, upper bound: 0.0012564
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.69
Output dim: 0, lower bound: -0.0012571, upper bound: 0.0012574
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.69
Output dim: 0, lower bound: -0.0012571, upper bound: 0.0012565
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.69
Output dim: 0, lower bound: -0.0012566, upper bound: 0.0012591
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.69
Output dim: 0, lower bound: -0.0012567, upper bound: 0.0012590

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 3, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012349, upper bound: 0.0012334
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012348, upper bound: 0.0012337
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 3, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012349, upper bound: 0.0012334
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012348, upper bound: 0.0012335
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 3, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012334, upper bound: 0.0012334
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012334, upper bound: 0.0012339
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 3, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012334, upper bound: 0.0012334
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012334, upper bound: 0.0012339
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 3, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012334, upper bound: 0.0012334
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012334, upper bound: 0.0012337
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 3, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012334, upper bound: 0.0012334
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012334, upper bound: 0.0012335
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 3, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012334, upper bound: 0.0012334
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012334, upper bound: 0.0012348
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 3, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012334, upper bound: 0.0012334
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012334, upper bound: 0.0012348
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 3, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012348, upper bound: 0.0012334
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012334, upper bound: 0.0012334
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 3, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012348, upper bound: 0.0012334
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012334, upper bound: 0.0012334
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012335, upper bound: 0.0012334
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012334, upper bound: 0.0012334
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012337, upper bound: 0.0012334
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012334, upper bound: 0.0012334
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012339, upper bound: 0.0012334
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012334, upper bound: 0.0012334
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012339, upper bound: 0.0012334
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012334, upper bound: 0.0012334
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012335, upper bound: 0.0012348
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012334, upper bound: 0.0012349
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012337, upper bound: 0.0012348
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012334, upper bound: 0.0012349
time: 0.18 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 1.40 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.40
Output dim: 0, lower bound: -0.0012349, upper bound: 0.0012334
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.40
Output dim: 0, lower bound: -0.0012348, upper bound: 0.0012337
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.40
Output dim: 0, lower bound: -0.0012349, upper bound: 0.0012334
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.40
Output dim: 0, lower bound: -0.0012348, upper bound: 0.0012335
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.40
Output dim: 0, lower bound: -0.0012334, upper bound: 0.0012334
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.40
Output dim: 0, lower bound: -0.0012334, upper bound: 0.0012339
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.40
Output dim: 0, lower bound: -0.0012334, upper bound: 0.0012334
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.40
Output dim: 0, lower bound: -0.0012334, upper bound: 0.0012339
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.40
Output dim: 0, lower bound: -0.0012334, upper bound: 0.0012334
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.40
Output dim: 0, lower bound: -0.0012334, upper bound: 0.0012337
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.40
Output dim: 0, lower bound: -0.0012334, upper bound: 0.0012334
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.40
Output dim: 0, lower bound: -0.0012334, upper bound: 0.0012335
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.40
Output dim: 0, lower bound: -0.0012334, upper bound: 0.0012334
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.40
Output dim: 0, lower bound: -0.0012334, upper bound: 0.0012348
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.40
Output dim: 0, lower bound: -0.0012334, upper bound: 0.0012334
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.40
Output dim: 0, lower bound: -0.0012334, upper bound: 0.0012348
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.40
Output dim: 0, lower bound: -0.0012348, upper bound: 0.0012334
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.40
Output dim: 0, lower bound: -0.0012334, upper bound: 0.0012334
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.40
Output dim: 0, lower bound: -0.0012348, upper bound: 0.0012334
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.40
Output dim: 0, lower bound: -0.0012334, upper bound: 0.0012334
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.40
Output dim: 0, lower bound: -0.0012335, upper bound: 0.0012334
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.40
Output dim: 0, lower bound: -0.0012334, upper bound: 0.0012334
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.40
Output dim: 0, lower bound: -0.0012337, upper bound: 0.0012334
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.40
Output dim: 0, lower bound: -0.0012334, upper bound: 0.0012334
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.40
Output dim: 0, lower bound: -0.0012339, upper bound: 0.0012334
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.40
Output dim: 0, lower bound: -0.0012334, upper bound: 0.0012334
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.40
Output dim: 0, lower bound: -0.0012339, upper bound: 0.0012334
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.40
Output dim: 0, lower bound: -0.0012334, upper bound: 0.0012334
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.40
Output dim: 0, lower bound: -0.0012335, upper bound: 0.0012348
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.40
Output dim: 0, lower bound: -0.0012334, upper bound: 0.0012349
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.40
Output dim: 0, lower bound: -0.0012337, upper bound: 0.0012348
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.40
Output dim: 0, lower bound: -0.0012334, upper bound: 0.0012349

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012240, upper bound: 0.0012226
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012242, upper bound: 0.0012226
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012228, upper bound: 0.0012226
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012242, upper bound: 0.0012226
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012240, upper bound: 0.0012226
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012242, upper bound: 0.0012226
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012228, upper bound: 0.0012226
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012242, upper bound: 0.0012226
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012227, upper bound: 0.0012226
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012227, upper bound: 0.0012226
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012228, upper bound: 0.0012226
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012228, upper bound: 0.0012226
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012228, upper bound: 0.0012226
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012228, upper bound: 0.0012226
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012228, upper bound: 0.0012226
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012228, upper bound: 0.0012226
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012228
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012228
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012228
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012228
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012228
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012228
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012227
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012227
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012242
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012228
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012242
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012240
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012242
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012228
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012242
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012240
time: 0.18 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 1.48 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012240, upper bound: 0.0012226
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012242, upper bound: 0.0012226
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012228, upper bound: 0.0012226
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012242, upper bound: 0.0012226
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012240, upper bound: 0.0012226
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012242, upper bound: 0.0012226
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012228, upper bound: 0.0012226
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012242, upper bound: 0.0012226
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012227, upper bound: 0.0012226
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012227, upper bound: 0.0012226
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012228, upper bound: 0.0012226
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012228, upper bound: 0.0012226
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012228, upper bound: 0.0012226
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012228, upper bound: 0.0012226
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012228, upper bound: 0.0012226
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012228, upper bound: 0.0012226
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012226
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012228
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012228
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012228
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012228
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012228
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012228
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012227
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012227
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012242
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012228
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012242
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012240
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012242
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012228
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012242
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0012226, upper bound: 0.0012240

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012234, upper bound: 0.0012220
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012235, upper bound: 0.0012220
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012222, upper bound: 0.0012220
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012235, upper bound: 0.0012220
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012234, upper bound: 0.0012220
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012235, upper bound: 0.0012220
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012222, upper bound: 0.0012220
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012235, upper bound: 0.0012220
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012221, upper bound: 0.0012220
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012221, upper bound: 0.0012220
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012221, upper bound: 0.0012220
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012221, upper bound: 0.0012220
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012221, upper bound: 0.0012220
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012221, upper bound: 0.0012220
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012221, upper bound: 0.0012220
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012221, upper bound: 0.0012220
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012221
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012221
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012221
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012221
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012221
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012221
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012221
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012221
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012235
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.52 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012222
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012235
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012234
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012235
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012222
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012235
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 47

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 30
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012234
time: 0.19 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 2.12 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012234, upper bound: 0.0012220
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012235, upper bound: 0.0012220
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012222, upper bound: 0.0012220
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012235, upper bound: 0.0012220
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012234, upper bound: 0.0012220
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012235, upper bound: 0.0012220
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012222, upper bound: 0.0012220
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012235, upper bound: 0.0012220
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012221, upper bound: 0.0012220
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012221, upper bound: 0.0012220
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012221, upper bound: 0.0012220
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012221, upper bound: 0.0012220
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012221, upper bound: 0.0012220
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012221, upper bound: 0.0012220
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012221, upper bound: 0.0012220
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012221, upper bound: 0.0012220
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012221
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012221
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012221
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012221
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012221
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012221
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012221
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012221
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012235
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012222
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012235
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012234
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012235
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012222
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012235
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012234

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012169, upper bound: 0.0012157
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012170, upper bound: 0.0012157
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012169, upper bound: 0.0012157
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012170, upper bound: 0.0012157
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012166, upper bound: 0.0012157
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012166, upper bound: 0.0012157
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012166, upper bound: 0.0012157
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012166, upper bound: 0.0012157
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012169, upper bound: 0.0012157
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012170, upper bound: 0.0012157
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012169, upper bound: 0.0012157
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012170, upper bound: 0.0012157
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012166, upper bound: 0.0012157
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012166, upper bound: 0.0012157
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012166, upper bound: 0.0012157
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012166, upper bound: 0.0012157
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012159, upper bound: 0.0012157
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012159, upper bound: 0.0012157
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012159, upper bound: 0.0012157
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012159, upper bound: 0.0012157
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012160, upper bound: 0.0012157
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012160, upper bound: 0.0012157
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012160, upper bound: 0.0012157
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012160, upper bound: 0.0012157
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012162, upper bound: 0.0012157
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012162, upper bound: 0.0012157
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012162, upper bound: 0.0012157
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012162, upper bound: 0.0012157
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012160, upper bound: 0.0012157
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012160, upper bound: 0.0012157
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012160, upper bound: 0.0012157
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012160, upper bound: 0.0012157
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012158
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012158
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012158
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012158
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557
1: -0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170
2: -0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386
3: -0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066
4: -0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 11
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 1
type: DSZ, layer: 5, pos: 7
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 43
type: DSZ, layer: 5, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012158
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012158
time: 0.20 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 3.12 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012169, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012170, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012169, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012170, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012166, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012166, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012166, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012166, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012169, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012170, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012169, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012170, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012166, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012166, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012166, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012166, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012159, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012159, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012159, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012159, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012160, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012160, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012160, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012160, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012162, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012162, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012162, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012162, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012160, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012160, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012160, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012160, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012158, upper bound: 0.0012157
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012158
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012158
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012158
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012158
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012158
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.12
Output dim: 0, lower bound: -0.0012157, upper bound: 0.0012158
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012221
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012221
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012221
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012221
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012221
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012221
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012221
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012221
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012235
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012222
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012235
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012234
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012235
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012222
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012235
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.12
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012234

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 1.57 + 418.65 = 420.22 seconds
