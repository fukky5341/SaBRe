## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_2.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 3)
Time budget: 420 seconds
Split limit: 100
Threshold: 0.5360177700000001


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713)
1: (-0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706)
2: (-0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765)
3: (-0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211)
4: (-0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.47 + 0.91 = 2.38 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.5955753, upper bound: 0.5955753

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5790429, upper bound: 0.5791239
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5791239, upper bound: 0.5790429
time: 0.27 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.67 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.67
Output dim: 0, lower bound: -0.5790429, upper bound: 0.5791239
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.67
Output dim: 0, lower bound: -0.5791239, upper bound: 0.5790429

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5734326, upper bound: 0.5790939
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5786512, upper bound: 0.5734326
time: 0.27 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5734326, upper bound: 0.5786512
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5790939, upper bound: 0.5734326
time: 0.28 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.00 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.00
Output dim: 0, lower bound: -0.5734326, upper bound: 0.5790939
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.00
Output dim: 0, lower bound: -0.5786512, upper bound: 0.5734326
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.00
Output dim: 0, lower bound: -0.5734326, upper bound: 0.5786512
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.00
Output dim: 0, lower bound: -0.5790939, upper bound: 0.5734326

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5733884, upper bound: 0.5790847
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5726981, upper bound: 0.5789347
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5786050, upper bound: 0.5728079
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5758523, upper bound: 0.5733884
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5733884, upper bound: 0.5758523
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5728079, upper bound: 0.5786050
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5789347, upper bound: 0.5726981
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5790847, upper bound: 0.5733884
time: 0.28 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.60 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.60
Output dim: 0, lower bound: -0.5733884, upper bound: 0.5790847
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.60
Output dim: 0, lower bound: -0.5726981, upper bound: 0.5789347
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.60
Output dim: 0, lower bound: -0.5786050, upper bound: 0.5728079
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.60
Output dim: 0, lower bound: -0.5758523, upper bound: 0.5733884
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.60
Output dim: 0, lower bound: -0.5733884, upper bound: 0.5758523
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.60
Output dim: 0, lower bound: -0.5728079, upper bound: 0.5786050
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.60
Output dim: 0, lower bound: -0.5789347, upper bound: 0.5726981
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.60
Output dim: 0, lower bound: -0.5790847, upper bound: 0.5733884

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.51 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5671951, upper bound: 0.5656427
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5625485, upper bound: 0.5724044
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.52 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5667363, upper bound: 0.5653670
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5617259, upper bound: 0.5722339
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.51 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5720106, upper bound: 0.5621209
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5651218, upper bound: 0.5667788
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.51 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5699436, upper bound: 0.5625257
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5633721, upper bound: 0.5671331
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.51 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5671331, upper bound: 0.5633721
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5625257, upper bound: 0.5699436
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.51 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5667788, upper bound: 0.5651218
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5621209, upper bound: 0.5720106
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.52 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5722339, upper bound: 0.5617259
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5653670, upper bound: 0.5667363
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.51 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5724044, upper bound: 0.5625485
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5656427, upper bound: 0.5671951
time: 0.30 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 3.10 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 0, lower bound: -0.5671951, upper bound: 0.5656427
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 0, lower bound: -0.5625485, upper bound: 0.5724044
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 0, lower bound: -0.5667363, upper bound: 0.5653670
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 0, lower bound: -0.5617259, upper bound: 0.5722339
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 0, lower bound: -0.5720106, upper bound: 0.5621209
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 0, lower bound: -0.5651218, upper bound: 0.5667788
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 0, lower bound: -0.5699436, upper bound: 0.5625257
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 0, lower bound: -0.5633721, upper bound: 0.5671331
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 0, lower bound: -0.5671331, upper bound: 0.5633721
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 0, lower bound: -0.5625257, upper bound: 0.5699436
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 0, lower bound: -0.5667788, upper bound: 0.5651218
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 0, lower bound: -0.5621209, upper bound: 0.5720106
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 0, lower bound: -0.5722339, upper bound: 0.5617259
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 0, lower bound: -0.5653670, upper bound: 0.5667363
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 0, lower bound: -0.5724044, upper bound: 0.5625485
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 0, lower bound: -0.5656427, upper bound: 0.5671951

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5665788, upper bound: 0.5651762
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5668219, upper bound: 0.5649462
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5619006, upper bound: 0.5719206
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5620105, upper bound: 0.5709518
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5660682, upper bound: 0.5648999
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5663614, upper bound: 0.5647121
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5611208, upper bound: 0.5717490
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5612060, upper bound: 0.5709520
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5709072, upper bound: 0.5615749
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5715267, upper bound: 0.5614610
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5645035, upper bound: 0.5664112
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5646569, upper bound: 0.5660682
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5688698, upper bound: 0.5619855
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5694851, upper bound: 0.5618591
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5628389, upper bound: 0.5667552
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5629293, upper bound: 0.5665366
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5665366, upper bound: 0.5629293
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5667552, upper bound: 0.5628389
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5618591, upper bound: 0.5694851
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5619855, upper bound: 0.5688698
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5660682, upper bound: 0.5646569
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664112, upper bound: 0.5645035
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5614610, upper bound: 0.5715267
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5615749, upper bound: 0.5709072
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5709520, upper bound: 0.5612060
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5717490, upper bound: 0.5611208
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5647121, upper bound: 0.5663614
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5648999, upper bound: 0.5660682
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5709518, upper bound: 0.5620105
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5719206, upper bound: 0.5619006
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5649462, upper bound: 0.5668219
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5651762, upper bound: 0.5665788
time: 0.28 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 2.71 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.5665788, upper bound: 0.5651762
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.5668219, upper bound: 0.5649462
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.5619006, upper bound: 0.5719206
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.5620105, upper bound: 0.5709518
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.5660682, upper bound: 0.5648999
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.5663614, upper bound: 0.5647121
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.5611208, upper bound: 0.5717490
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.5612060, upper bound: 0.5709520
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.5709072, upper bound: 0.5615749
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.5715267, upper bound: 0.5614610
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.5645035, upper bound: 0.5664112
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.5646569, upper bound: 0.5660682
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.5688698, upper bound: 0.5619855
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.5694851, upper bound: 0.5618591
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.5628389, upper bound: 0.5667552
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.5629293, upper bound: 0.5665366
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.5665366, upper bound: 0.5629293
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.5667552, upper bound: 0.5628389
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.5618591, upper bound: 0.5694851
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.5619855, upper bound: 0.5688698
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.5660682, upper bound: 0.5646569
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.5664112, upper bound: 0.5645035
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.5614610, upper bound: 0.5715267
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.5615749, upper bound: 0.5709072
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.5709520, upper bound: 0.5612060
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.5717490, upper bound: 0.5611208
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.5647121, upper bound: 0.5663614
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.5648999, upper bound: 0.5660682
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.5709518, upper bound: 0.5620105
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.5719206, upper bound: 0.5619006
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.5649462, upper bound: 0.5668219
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.71
Output dim: 0, lower bound: -0.5651762, upper bound: 0.5665788

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5599650, upper bound: 0.5625812
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5639677, upper bound: 0.5627688
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5649559, upper bound: 0.5608876
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5647019, upper bound: 0.5626887
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5599650, upper bound: 0.5693735
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5589197, upper bound: 0.5694639
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5599957, upper bound: 0.5661431
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5596253, upper bound: 0.5686997
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5642110, upper bound: 0.5620401
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5639828, upper bound: 0.5625029
time: 0.36 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5642359, upper bound: 0.5601517
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5642348, upper bound: 0.5624372
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5590364, upper bound: 0.5691269
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5589182, upper bound: 0.5692993
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5590605, upper bound: 0.5660002
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5590609, upper bound: 0.5686744
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5686191, upper bound: 0.5594093
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5659818, upper bound: 0.5595791
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5690700, upper bound: 0.5589194
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5686175, upper bound: 0.5595507
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5621974, upper bound: 0.5643292
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5600324, upper bound: 0.5643496
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5622519, upper bound: 0.5639828
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5615304, upper bound: 0.5642250
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5664667, upper bound: 0.5596253
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5659861, upper bound: 0.5599957
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5665798, upper bound: 0.5589197
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5666678, upper bound: 0.5599650
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5601693, upper bound: 0.5645994
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5600144, upper bound: 0.5649344
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5601878, upper bound: 0.5639306
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5602052, upper bound: 0.5647873
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5647873, upper bound: 0.5602052
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5639306, upper bound: 0.5601878
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5649344, upper bound: 0.5600144
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5645994, upper bound: 0.5601693
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5599650, upper bound: 0.5666678
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5589197, upper bound: 0.5665798
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5599957, upper bound: 0.5659861
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5596253, upper bound: 0.5664667
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5642250, upper bound: 0.5615304
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5639828, upper bound: 0.5622519
time: 0.35 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5643496, upper bound: 0.5600324
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5643292, upper bound: 0.5621974
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5595507, upper bound: 0.5686175
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5589194, upper bound: 0.5690700
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5595791, upper bound: 0.5659818
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5594093, upper bound: 0.5686191
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5686744, upper bound: 0.5590609
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5660002, upper bound: 0.5590605
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5692993, upper bound: 0.5589182
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5691269, upper bound: 0.5590364
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5624372, upper bound: 0.5642348
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5601517, upper bound: 0.5642359
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5625029, upper bound: 0.5639828
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5620401, upper bound: 0.5642110
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5686997, upper bound: 0.5596253
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5661431, upper bound: 0.5599957
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5694639, upper bound: 0.5589197
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5693735, upper bound: 0.5599650
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5626887, upper bound: 0.5647019
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5608876, upper bound: 0.5649559
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5627688, upper bound: 0.5639677
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5625812, upper bound: 0.5647874
time: 0.29 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.18 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5599650, upper bound: 0.5625812
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5639677, upper bound: 0.5627688
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5649559, upper bound: 0.5608876
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5647019, upper bound: 0.5626887
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5599650, upper bound: 0.5693735
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5589197, upper bound: 0.5694639
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5599957, upper bound: 0.5661431
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5596253, upper bound: 0.5686997
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5642110, upper bound: 0.5620401
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5639828, upper bound: 0.5625029
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5642359, upper bound: 0.5601517
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5642348, upper bound: 0.5624372
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5590364, upper bound: 0.5691269
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5589182, upper bound: 0.5692993
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5590605, upper bound: 0.5660002
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5590609, upper bound: 0.5686744
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5686191, upper bound: 0.5594093
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5659818, upper bound: 0.5595791
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5690700, upper bound: 0.5589194
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5686175, upper bound: 0.5595507
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5621974, upper bound: 0.5643292
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5600324, upper bound: 0.5643496
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5622519, upper bound: 0.5639828
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5615304, upper bound: 0.5642250
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5664667, upper bound: 0.5596253
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5659861, upper bound: 0.5599957
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5665798, upper bound: 0.5589197
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5666678, upper bound: 0.5599650
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5601693, upper bound: 0.5645994
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5600144, upper bound: 0.5649344
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5601878, upper bound: 0.5639306
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5602052, upper bound: 0.5647873
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5647873, upper bound: 0.5602052
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5639306, upper bound: 0.5601878
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5649344, upper bound: 0.5600144
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5645994, upper bound: 0.5601693
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5599650, upper bound: 0.5666678
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5589197, upper bound: 0.5665798
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5599957, upper bound: 0.5659861
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5596253, upper bound: 0.5664667
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5642250, upper bound: 0.5615304
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5639828, upper bound: 0.5622519
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5643496, upper bound: 0.5600324
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5643292, upper bound: 0.5621974
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5595507, upper bound: 0.5686175
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5589194, upper bound: 0.5690700
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5595791, upper bound: 0.5659818
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5594093, upper bound: 0.5686191
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5686744, upper bound: 0.5590609
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5660002, upper bound: 0.5590605
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5692993, upper bound: 0.5589182
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5691269, upper bound: 0.5590364
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5624372, upper bound: 0.5642348
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5601517, upper bound: 0.5642359
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5625029, upper bound: 0.5639828
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5620401, upper bound: 0.5642110
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5686997, upper bound: 0.5596253
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5661431, upper bound: 0.5599957
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5694639, upper bound: 0.5589197
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5693735, upper bound: 0.5599650
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5626887, upper bound: 0.5647019
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5608876, upper bound: 0.5649559
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5627688, upper bound: 0.5639677
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -0.5625812, upper bound: 0.5647874

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5565789, upper bound: 0.5571878
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5621914, upper bound: 0.5568063
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5570593, upper bound: 0.5575911
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5596963, upper bound: 0.5572690
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5563367, upper bound: 0.5554409
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5623092, upper bound: 0.5553039
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5570005, upper bound: 0.5575112
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5617448, upper bound: 0.5572170
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5545324, upper bound: 0.5647447
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5565916, upper bound: 0.5627782
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5539240, upper bound: 0.5648851
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5544446, upper bound: 0.5633418
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5544686, upper bound: 0.5611999
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5566269, upper bound: 0.5606089
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5541453, upper bound: 0.5642114
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5556464, upper bound: 0.5627991
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5552978, upper bound: 0.5566776
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5600257, upper bound: 0.5562604
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5560203, upper bound: 0.5573272
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5596891, upper bound: 0.5569909
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5538839, upper bound: 0.5546579
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5601535, upper bound: 0.5545527
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5559356, upper bound: 0.5572612
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5602137, upper bound: 0.5569522
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5539607, upper bound: 0.5643161
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5546274, upper bound: 0.5623900
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5538666, upper bound: 0.5646653
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5544283, upper bound: 0.5631676
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5538839, upper bound: 0.5609643
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5546468, upper bound: 0.5605214
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5538675, upper bound: 0.5641572
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5546484, upper bound: 0.5628746
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5627889, upper bound: 0.5560743
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5639889, upper bound: 0.5537898
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5605614, upper bound: 0.5566609
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5609203, upper bound: 0.5537958
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5629327, upper bound: 0.5548503
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5643150, upper bound: 0.5537868
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5620484, upper bound: 0.5566158
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5636107, upper bound: 0.5538958
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5566815, upper bound: 0.5615977
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5569683, upper bound: 0.5549580
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5544015, upper bound: 0.5619062
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5545240, upper bound: 0.5538247
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5567132, upper bound: 0.5596958
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5570210, upper bound: 0.5550754
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5556960, upper bound: 0.5616347
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5560655, upper bound: 0.5544108
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5611525, upper bound: 0.5559270
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5613346, upper bound: 0.5532326
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5605942, upper bound: 0.5566684
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5609091, upper bound: 0.5534205
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5614244, upper bound: 0.5544524
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5615634, upper bound: 0.5532055
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.51 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5546454, upper bound: 0.5566258
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5616490, upper bound: 0.5536373
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.51 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5546319, upper bound: 0.5617515
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5546846, upper bound: 0.5526663
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.52 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5543906, upper bound: 0.5623092
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5544728, upper bound: 0.5528321
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.51 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5546550, upper bound: 0.5596963
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5547140, upper bound: 0.5526784
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.52 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5546454, upper bound: 0.5621914
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5547148, upper bound: 0.5531114
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.52 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5531114, upper bound: 0.5547148
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5621914, upper bound: 0.5546454
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.52 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5526784, upper bound: 0.5547140
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5596963, upper bound: 0.5546550
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.53 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5528321, upper bound: 0.5544728
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5623092, upper bound: 0.5543906
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.53 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5526663, upper bound: 0.5546846
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5617515, upper bound: 0.5546319
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.53 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5536373, upper bound: 0.5616490
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5566258, upper bound: 0.5614240
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.53 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5532055, upper bound: 0.5615634
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5544524, upper bound: 0.5614244
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5534205, upper bound: 0.5609091
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5566684, upper bound: 0.5605942
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5532326, upper bound: 0.5613346
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5559270, upper bound: 0.5611525
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5538958, upper bound: 0.5560655
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5616347, upper bound: 0.5556960
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5550754, upper bound: 0.5570210
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5596958, upper bound: 0.5567132
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5538247, upper bound: 0.5545240
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5619062, upper bound: 0.5544015
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5549580, upper bound: 0.5569683
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5615977, upper bound: 0.5566815
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5538958, upper bound: 0.5636107
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5566158, upper bound: 0.5620484
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.57 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5537868, upper bound: 0.5643150
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5548503, upper bound: 0.5629327
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.57 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5537958, upper bound: 0.5609203
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5566609, upper bound: 0.5605614
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.57 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5537898, upper bound: 0.5639889
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5560743, upper bound: 0.5627889
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5628746, upper bound: 0.5546484
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5641572, upper bound: 0.5538675
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5605214, upper bound: 0.5546468
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5609643, upper bound: 0.5538839
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5631676, upper bound: 0.5544283
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5646653, upper bound: 0.5538666
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5623900, upper bound: 0.5546274
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5643161, upper bound: 0.5539607
time: 0.33 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5569522, upper bound: 0.5602137
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5572612, upper bound: 0.5559356
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5545527, upper bound: 0.5601535
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5546579, upper bound: 0.5541537
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5569909, upper bound: 0.5596891
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5573272, upper bound: 0.5560203
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5562604, upper bound: 0.5600257
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5566776, upper bound: 0.5552978
time: 0.34 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5627991, upper bound: 0.5556464
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5642114, upper bound: 0.5541453
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5606089, upper bound: 0.5566269
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5611999, upper bound: 0.5544686
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5633418, upper bound: 0.5544446
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5648851, upper bound: 0.5539240
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5627782, upper bound: 0.5565916
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5647447, upper bound: 0.5545324
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5572170, upper bound: 0.5617448
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5575112, upper bound: 0.5570005
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5553039, upper bound: 0.5623092
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5554409, upper bound: 0.5563367
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5572690, upper bound: 0.5596963
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5575911, upper bound: 0.5570593
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 3, pos: 43

### Candidate
type: DSZ, layer: 3, pos: 22

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 17

### Candidate
type: DSZ, layer: 3, pos: 13

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5568063, upper bound: 0.5621914
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5571878, upper bound: 0.5565789
time: 0.32 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 2.53 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5565789, upper bound: 0.5571878
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5621914, upper bound: 0.5568063
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5570593, upper bound: 0.5575911
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5596963, upper bound: 0.5572690
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5563367, upper bound: 0.5554409
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5623092, upper bound: 0.5553039
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5570005, upper bound: 0.5575112
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5617448, upper bound: 0.5572170
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5545324, upper bound: 0.5647447
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5565916, upper bound: 0.5627782
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5539240, upper bound: 0.5648851
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5544446, upper bound: 0.5633418
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5544686, upper bound: 0.5611999
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5566269, upper bound: 0.5606089
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5541453, upper bound: 0.5642114
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5556464, upper bound: 0.5627991
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5552978, upper bound: 0.5566776
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5600257, upper bound: 0.5562604
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5560203, upper bound: 0.5573272
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5596891, upper bound: 0.5569909
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5538839, upper bound: 0.5546579
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5601535, upper bound: 0.5545527
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5559356, upper bound: 0.5572612
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5602137, upper bound: 0.5569522
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5539607, upper bound: 0.5643161
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5546274, upper bound: 0.5623900
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5538666, upper bound: 0.5646653
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5544283, upper bound: 0.5631676
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5538839, upper bound: 0.5609643
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5546468, upper bound: 0.5605214
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5538675, upper bound: 0.5641572
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5546484, upper bound: 0.5628746
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5627889, upper bound: 0.5560743
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5639889, upper bound: 0.5537898
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5605614, upper bound: 0.5566609
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5609203, upper bound: 0.5537958
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5629327, upper bound: 0.5548503
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5643150, upper bound: 0.5537868
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5620484, upper bound: 0.5566158
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5636107, upper bound: 0.5538958
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5566815, upper bound: 0.5615977
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5569683, upper bound: 0.5549580
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5544015, upper bound: 0.5619062
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5545240, upper bound: 0.5538247
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5567132, upper bound: 0.5596958
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5570210, upper bound: 0.5550754
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5556960, upper bound: 0.5616347
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5560655, upper bound: 0.5544108
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5611525, upper bound: 0.5559270
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5613346, upper bound: 0.5532326
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5605942, upper bound: 0.5566684
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5609091, upper bound: 0.5534205
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5614244, upper bound: 0.5544524
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5615634, upper bound: 0.5532055
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5546454, upper bound: 0.5566258
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5616490, upper bound: 0.5536373
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5546319, upper bound: 0.5617515
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5546846, upper bound: 0.5526663
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5543906, upper bound: 0.5623092
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5544728, upper bound: 0.5528321
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5546550, upper bound: 0.5596963
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5547140, upper bound: 0.5526784
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5546454, upper bound: 0.5621914
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5547148, upper bound: 0.5531114
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5531114, upper bound: 0.5547148
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5621914, upper bound: 0.5546454
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5526784, upper bound: 0.5547140
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5596963, upper bound: 0.5546550
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5528321, upper bound: 0.5544728
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5623092, upper bound: 0.5543906
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5526663, upper bound: 0.5546846
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5617515, upper bound: 0.5546319
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5536373, upper bound: 0.5616490
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5566258, upper bound: 0.5614240
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5532055, upper bound: 0.5615634
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5544524, upper bound: 0.5614244
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5534205, upper bound: 0.5609091
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5566684, upper bound: 0.5605942
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5532326, upper bound: 0.5613346
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5559270, upper bound: 0.5611525
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5538958, upper bound: 0.5560655
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5616347, upper bound: 0.5556960
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5550754, upper bound: 0.5570210
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5596958, upper bound: 0.5567132
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5538247, upper bound: 0.5545240
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5619062, upper bound: 0.5544015
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5549580, upper bound: 0.5569683
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5615977, upper bound: 0.5566815
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5538958, upper bound: 0.5636107
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5566158, upper bound: 0.5620484
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5537868, upper bound: 0.5643150
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5548503, upper bound: 0.5629327
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5537958, upper bound: 0.5609203
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5566609, upper bound: 0.5605614
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5537898, upper bound: 0.5639889
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5560743, upper bound: 0.5627889
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5628746, upper bound: 0.5546484
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5641572, upper bound: 0.5538675
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5605214, upper bound: 0.5546468
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5609643, upper bound: 0.5538839
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5631676, upper bound: 0.5544283
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5646653, upper bound: 0.5538666
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5623900, upper bound: 0.5546274
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5643161, upper bound: 0.5539607
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5569522, upper bound: 0.5602137
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5572612, upper bound: 0.5559356
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5545527, upper bound: 0.5601535
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5546579, upper bound: 0.5541537
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5569909, upper bound: 0.5596891
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5573272, upper bound: 0.5560203
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5562604, upper bound: 0.5600257
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5566776, upper bound: 0.5552978
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5627991, upper bound: 0.5556464
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5642114, upper bound: 0.5541453
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5606089, upper bound: 0.5566269
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5611999, upper bound: 0.5544686
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5633418, upper bound: 0.5544446
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5648851, upper bound: 0.5539240
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5627782, upper bound: 0.5565916
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5647447, upper bound: 0.5545324
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5572170, upper bound: 0.5617448
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5575112, upper bound: 0.5570005
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5553039, upper bound: 0.5623092
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5554409, upper bound: 0.5563367
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5572690, upper bound: 0.5596963
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5575911, upper bound: 0.5570593
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5568063, upper bound: 0.5621914
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.5571878, upper bound: 0.5565789

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5498505, upper bound: 0.5531639
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5523159, upper bound: 0.5512543
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.52 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5509433, upper bound: 0.5526725
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5582589, upper bound: 0.5510164
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5493063, upper bound: 0.5536115
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5527307, upper bound: 0.5522790
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.52 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5495892, upper bound: 0.5531743
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5557213, upper bound: 0.5519660
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.53 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5495336, upper bound: 0.5513243
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5521035, upper bound: 0.5506778
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.53 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5495336, upper bound: 0.5510355
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5584168, upper bound: 0.5504746
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5493166, upper bound: 0.5535323
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5526694, upper bound: 0.5523712
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5495904, upper bound: 0.5530695
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5578192, upper bound: 0.5520230
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5499740, upper bound: 0.5607204
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5501584, upper bound: 0.5505016
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5518116, upper bound: 0.5584235
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5526278, upper bound: 0.5503517
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5483949, upper bound: 0.5609055
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5495288, upper bound: 0.5517108
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.57 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5497102, upper bound: 0.5589448
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5503102, upper bound: 0.5515279
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.57 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5497966, upper bound: 0.5571406
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5501164, upper bound: 0.5504568
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.57 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5515973, upper bound: 0.5563130
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5526624, upper bound: 0.5502785
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5493839, upper bound: 0.5602633
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5498244, upper bound: 0.5517971
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5502299, upper bound: 0.5584477
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5516471, upper bound: 0.5515923
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5495066, upper bound: 0.5526244
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5509985, upper bound: 0.5507750
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.76 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5498338, upper bound: 0.5520886
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5560243, upper bound: 0.5504976
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.69 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5493050, upper bound: 0.5533213
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5516914, upper bound: 0.5519811
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5495749, upper bound: 0.5528672
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5557129, upper bound: 0.5516562
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5493064, upper bound: 0.5504411
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5499415, upper bound: 0.5497962
time: 0.34 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5496553, upper bound: 0.5502130
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5561828, upper bound: 0.5496739
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5493054, upper bound: 0.5532560
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5516042, upper bound: 0.5521353
time: 0.34 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5495707, upper bound: 0.5527848
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5562440, upper bound: 0.5517693
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5494999, upper bound: 0.5602677
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5495508, upper bound: 0.5500349
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5499173, upper bound: 0.5580165
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5504075, upper bound: 0.5498688
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5492943, upper bound: 0.5606643
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5495004, upper bound: 0.5513974
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5496640, upper bound: 0.5587603
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5502076, upper bound: 0.5511960
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5493791, upper bound: 0.5569364
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5494983, upper bound: 0.5496678
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5498274, upper bound: 0.5561452
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5504516, upper bound: 0.5495763
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5493300, upper bound: 0.5602139
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5494998, upper bound: 0.5515766
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5497577, upper bound: 0.5584706
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5504608, upper bound: 0.5513581
time: 0.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5511073, upper bound: 0.5520727
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5583923, upper bound: 0.5507441
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5513108, upper bound: 0.5493706
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5600239, upper bound: 0.5492962
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5493063, upper bound: 0.5526740
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5562501, upper bound: 0.5516605
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5495728, upper bound: 0.5493678
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5569036, upper bound: 0.5493117
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5508435, upper bound: 0.5508972
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5585115, upper bound: 0.5504631
time: 0.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.66 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5510244, upper bound: 0.5493986
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5602719, upper bound: 0.5492943
time: 0.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5493166, upper bound: 0.5526289
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5576618, upper bound: 0.5517938
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.67 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5495507, upper bound: 0.5494505
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5595340, upper bound: 0.5494375
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.67 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5515065, upper bound: 0.5577074
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5524957, upper bound: 0.5501578
time: 0.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.66 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5518442, upper bound: 0.5506243
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5529260, upper bound: 0.5492528
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.67 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5495716, upper bound: 0.5579180
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5500594, upper bound: 0.5506227
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.67 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5496897, upper bound: 0.5494371
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5502868, upper bound: 0.5492291
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.68 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5513392, upper bound: 0.5557238
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5525608, upper bound: 0.5501578
time: 0.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.68 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5516213, upper bound: 0.5507443
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5529779, upper bound: 0.5492855
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.68 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 5
type: DSZ, layer: 5, pos: 6
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 13
type: DSZ, layer: 5, pos: 23
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 20
type: DSZ, layer: 5, pos: 29

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 5, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5499956, upper bound: 0.5576258
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5515132, upper bound: 0.5508454
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 1.70 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 2.38 + 417.95 = 420.33 seconds
