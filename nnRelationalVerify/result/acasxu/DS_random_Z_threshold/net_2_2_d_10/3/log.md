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
execution time: IAR + RelationalAnalysis = 0.64 + 0.83 = 1.47 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.5955753, upper bound: 0.5955753

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5939137, upper bound: 0.5906290
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5906290, upper bound: 0.5939137
time: 0.21 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.49 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.49
Output dim: 0, lower bound: -0.5939137, upper bound: 0.5906290
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.49
Output dim: 0, lower bound: -0.5906290, upper bound: 0.5939137

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 0.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5496833, upper bound: 0.5463336
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5496833, upper bound: 0.5420812
time: 0.24 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 0.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5420812, upper bound: 0.5496833
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5463336, upper bound: 0.5496833
time: 0.20 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.13 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.13
Output dim: 0, lower bound: -0.5496833, upper bound: 0.5463336
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.13
Output dim: 0, lower bound: -0.5496833, upper bound: 0.5420812
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.13
Output dim: 0, lower bound: -0.5420812, upper bound: 0.5496833
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.13
Output dim: 0, lower bound: -0.5463336, upper bound: 0.5496833

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 0.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5447122, upper bound: 0.5425910
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5433332, upper bound: 0.5433826
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 0.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5447122, upper bound: 0.5418963
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5433332, upper bound: 0.5418877
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 0.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5286841, upper bound: 0.5345450
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5306097, upper bound: 0.5345450
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 0.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5433826, upper bound: 0.5433332
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5425910, upper bound: 0.5447122
time: 0.22 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 1.17 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.17
Output dim: 0, lower bound: -0.5447122, upper bound: 0.5425910
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.17
Output dim: 0, lower bound: -0.5433332, upper bound: 0.5433826
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.17
Output dim: 0, lower bound: -0.5447122, upper bound: 0.5418963
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.17
Output dim: 0, lower bound: -0.5433332, upper bound: 0.5418877
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 1.17
Output dim: 0, lower bound: -0.5286841, upper bound: 0.5345450
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 1.17
Output dim: 0, lower bound: -0.5306097, upper bound: 0.5345450
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.17
Output dim: 0, lower bound: -0.5433826, upper bound: 0.5433332
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.17
Output dim: 0, lower bound: -0.5425910, upper bound: 0.5447122

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 0.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5324451, upper bound: 0.5307590
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5324451, upper bound: 0.5307590
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 0.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5318596, upper bound: 0.5312306
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5317633, upper bound: 0.5312306
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 0.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 32

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5324451, upper bound: 0.5301773
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5324451, upper bound: 0.5265152
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 32

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5318596, upper bound: 0.5301285
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5317633, upper bound: 0.5252658
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 0.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5312306, upper bound: 0.5317633
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5312306, upper bound: 0.5318596
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713
1: -0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706
2: -0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765
3: -0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211
4: -0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5307590, upper bound: 0.5324451
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5307590, upper bound: 0.5324451
time: 0.21 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 1.02 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 1.02
Output dim: 0, lower bound: -0.5324451, upper bound: 0.5307590
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 1.02
Output dim: 0, lower bound: -0.5324451, upper bound: 0.5307590
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 1.02
Output dim: 0, lower bound: -0.5318596, upper bound: 0.5312306
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 1.02
Output dim: 0, lower bound: -0.5317633, upper bound: 0.5312306
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 1.02
Output dim: 0, lower bound: -0.5324451, upper bound: 0.5301773
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 1.02
Output dim: 0, lower bound: -0.5324451, upper bound: 0.5265152
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 1.02
Output dim: 0, lower bound: -0.5318596, upper bound: 0.5301285
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 1.02
Output dim: 0, lower bound: -0.5317633, upper bound: 0.5252658
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 1.02
Output dim: 0, lower bound: -0.5312306, upper bound: 0.5317633
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 1.02
Output dim: 0, lower bound: -0.5312306, upper bound: 0.5318596
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 1.02
Output dim: 0, lower bound: -0.5307590, upper bound: 0.5324451
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 1.02
Output dim: 0, lower bound: -0.5307590, upper bound: 0.5324451

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 1.47 + 13.20 = 14.67 seconds
