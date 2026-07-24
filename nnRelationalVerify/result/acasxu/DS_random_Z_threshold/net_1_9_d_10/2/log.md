## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_9.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 2)
Time budget: 420 seconds
Split limit: 100
Threshold: 7.420799999999999e-05


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284)
1: (-0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442)
2: (-0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118)
3: (-0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389)
4: (-0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.03 + 0.55 = 1.58 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0000773, upper bound: 0.0000773

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000769, upper bound: 0.0000769
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000769, upper bound: 0.0000769
time: 0.18 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.37 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.37
Output dim: 0, lower bound: -0.0000769, upper bound: 0.0000769
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.37
Output dim: 0, lower bound: -0.0000769, upper bound: 0.0000769

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000753, upper bound: 0.0000768
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000768, upper bound: 0.0000753
time: 0.13 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000763, upper bound: 0.0000767
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000767, upper bound: 0.0000767
time: 0.13 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.20 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.20
Output dim: 0, lower bound: -0.0000753, upper bound: 0.0000768
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.20
Output dim: 0, lower bound: -0.0000768, upper bound: 0.0000753
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.20
Output dim: 0, lower bound: -0.0000763, upper bound: 0.0000767
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.20
Output dim: 0, lower bound: -0.0000767, upper bound: 0.0000767

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000752, upper bound: 0.0000766
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000752, upper bound: 0.0000767
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000759, upper bound: 0.0000748
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000762, upper bound: 0.0000748
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000756, upper bound: 0.0000760
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000760, upper bound: 0.0000749
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000766, upper bound: 0.0000764
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000766, upper bound: 0.0000766
time: 0.14 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 1.21 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.21
Output dim: 0, lower bound: -0.0000752, upper bound: 0.0000766
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.21
Output dim: 0, lower bound: -0.0000752, upper bound: 0.0000767
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.21
Output dim: 0, lower bound: -0.0000759, upper bound: 0.0000748
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.21
Output dim: 0, lower bound: -0.0000762, upper bound: 0.0000748
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.21
Output dim: 0, lower bound: -0.0000756, upper bound: 0.0000760
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.21
Output dim: 0, lower bound: -0.0000760, upper bound: 0.0000749
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.21
Output dim: 0, lower bound: -0.0000766, upper bound: 0.0000764
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.21
Output dim: 0, lower bound: -0.0000766, upper bound: 0.0000766

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000751, upper bound: 0.0000765
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000754
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000759
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000748, upper bound: 0.0000755
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000758, upper bound: 0.0000748
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000748
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000761, upper bound: 0.0000745
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000747
time: 0.14 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000756
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000746, upper bound: 0.0000759
time: 0.14 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000748, upper bound: 0.0000747
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000759, upper bound: 0.0000746
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000750, upper bound: 0.0000764
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000765, upper bound: 0.0000751
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000746, upper bound: 0.0000760
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000758
time: 0.14 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 1.25 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.25
Output dim: 0, lower bound: -0.0000751, upper bound: 0.0000765
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.25
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000754
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.25
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000759
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.25
Output dim: 0, lower bound: -0.0000748, upper bound: 0.0000755
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.25
Output dim: 0, lower bound: -0.0000758, upper bound: 0.0000748
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.25
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000748
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.25
Output dim: 0, lower bound: -0.0000761, upper bound: 0.0000745
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.25
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000747
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.25
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000756
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.25
Output dim: 0, lower bound: -0.0000746, upper bound: 0.0000759
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.25
Output dim: 0, lower bound: -0.0000748, upper bound: 0.0000747
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.25
Output dim: 0, lower bound: -0.0000759, upper bound: 0.0000746
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.25
Output dim: 0, lower bound: -0.0000750, upper bound: 0.0000764
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.25
Output dim: 0, lower bound: -0.0000765, upper bound: 0.0000751
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.25
Output dim: 0, lower bound: -0.0000746, upper bound: 0.0000760
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.25
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000758

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000755
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000744
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000745, upper bound: 0.0000751
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000746, upper bound: 0.0000744
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000758
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000744, upper bound: 0.0000758
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000749
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000754
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000757, upper bound: 0.0000748
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000747
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000746, upper bound: 0.0000747
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000744, upper bound: 0.0000747
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000760, upper bound: 0.0000744
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000758, upper bound: 0.0000745
time: 0.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000754, upper bound: 0.0000746
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000747
time: 0.14 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000755
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000754, upper bound: 0.0000748
time: 0.14 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000745, upper bound: 0.0000758
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000744, upper bound: 0.0000746
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000744
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000746
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000758, upper bound: 0.0000744
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000751, upper bound: 0.0000745
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000746, upper bound: 0.0000755
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000747
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000749, upper bound: 0.0000748
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000758, upper bound: 0.0000746
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000744, upper bound: 0.0000760
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000744, upper bound: 0.0000748
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000757
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000748
time: 0.16 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 1.46 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000755
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000744
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000745, upper bound: 0.0000751
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000746, upper bound: 0.0000744
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000758
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000744, upper bound: 0.0000758
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000749
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000754
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000757, upper bound: 0.0000748
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000747
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000746, upper bound: 0.0000747
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000744, upper bound: 0.0000747
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000760, upper bound: 0.0000744
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000758, upper bound: 0.0000745
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000754, upper bound: 0.0000746
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000747
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000755
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000754, upper bound: 0.0000748
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000745, upper bound: 0.0000758
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000744, upper bound: 0.0000746
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000744
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000746
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000758, upper bound: 0.0000744
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000751, upper bound: 0.0000745
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000746, upper bound: 0.0000755
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000747
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000749, upper bound: 0.0000748
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000758, upper bound: 0.0000746
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000744, upper bound: 0.0000760
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000744, upper bound: 0.0000748
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000757
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000748

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000743
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000741
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000734, upper bound: 0.0000732
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000734, upper bound: 0.0000732
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000750
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000741, upper bound: 0.0000749
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000739, upper bound: 0.0000737
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000737, upper bound: 0.0000737
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000739, upper bound: 0.0000753
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000753
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000737, upper bound: 0.0000742
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000740
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000734, upper bound: 0.0000734
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000734, upper bound: 0.0000734
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000746, upper bound: 0.0000752
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000744, upper bound: 0.0000749
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000752, upper bound: 0.0000740
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000753, upper bound: 0.0000740
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000732, upper bound: 0.0000734
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000732, upper bound: 0.0000734
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000737, upper bound: 0.0000740
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000738, upper bound: 0.0000740
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000745
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000741, upper bound: 0.0000746
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000732
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000732
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000753, upper bound: 0.0000738
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000752, upper bound: 0.0000738
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000754, upper bound: 0.0000743
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000754, upper bound: 0.0000744
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000753, upper bound: 0.0000745
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000753, upper bound: 0.0000745
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000743
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000742
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000749, upper bound: 0.0000740
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000749, upper bound: 0.0000741
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000738, upper bound: 0.0000752
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000738, upper bound: 0.0000753
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000741, upper bound: 0.0000741
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000741, upper bound: 0.0000744
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000737
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000737
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000738
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000739
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000753, upper bound: 0.0000737
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000753, upper bound: 0.0000737
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000733, upper bound: 0.0000732
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000734, upper bound: 0.0000732
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000738, upper bound: 0.0000743
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000738, upper bound: 0.0000742
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000740
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000739, upper bound: 0.0000740
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000738, upper bound: 0.0000741
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000739, upper bound: 0.0000741
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000752, upper bound: 0.0000741
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000756, upper bound: 0.0000745
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000737, upper bound: 0.0000742
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000740
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000741, upper bound: 0.0000745
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000741, upper bound: 0.0000746
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000734, upper bound: 0.0000743
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000734, upper bound: 0.0000743
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000733, upper bound: 0.0000734
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000734, upper bound: 0.0000735
time: 0.16 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 1.44 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000743
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000741
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000734, upper bound: 0.0000732
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000734, upper bound: 0.0000732
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000750
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000741, upper bound: 0.0000749
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000739, upper bound: 0.0000737
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000737, upper bound: 0.0000737
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000739, upper bound: 0.0000753
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000753
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000737, upper bound: 0.0000742
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000740
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000734, upper bound: 0.0000734
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000734, upper bound: 0.0000734
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000746, upper bound: 0.0000752
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000744, upper bound: 0.0000749
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000752, upper bound: 0.0000740
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000753, upper bound: 0.0000740
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000732, upper bound: 0.0000734
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000732, upper bound: 0.0000734
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000737, upper bound: 0.0000740
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000738, upper bound: 0.0000740
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000745
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000741, upper bound: 0.0000746
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000732
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000732
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000753, upper bound: 0.0000738
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000752, upper bound: 0.0000738
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000754, upper bound: 0.0000743
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000754, upper bound: 0.0000744
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000753, upper bound: 0.0000745
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000753, upper bound: 0.0000745
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000743
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000742
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000749, upper bound: 0.0000740
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000749, upper bound: 0.0000741
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000738, upper bound: 0.0000752
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000738, upper bound: 0.0000753
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000741, upper bound: 0.0000741
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000741, upper bound: 0.0000744
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000737
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000737
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000738
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000739
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000753, upper bound: 0.0000737
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000753, upper bound: 0.0000737
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000733, upper bound: 0.0000732
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000734, upper bound: 0.0000732
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000738, upper bound: 0.0000743
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000738, upper bound: 0.0000742
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000740
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000739, upper bound: 0.0000740
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000738, upper bound: 0.0000741
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000739, upper bound: 0.0000741
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000752, upper bound: 0.0000741
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000756, upper bound: 0.0000745
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000737, upper bound: 0.0000742
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000740
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000741, upper bound: 0.0000745
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000741, upper bound: 0.0000746
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000734, upper bound: 0.0000743
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000734, upper bound: 0.0000743
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000733, upper bound: 0.0000734
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.44
Output dim: 0, lower bound: -0.0000734, upper bound: 0.0000735

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000738
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000739
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000741
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000735, upper bound: 0.0000739
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000735, upper bound: 0.0000741
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000735, upper bound: 0.0000740
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000735, upper bound: 0.0000738
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000735, upper bound: 0.0000735
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000726, upper bound: 0.0000727
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000726, upper bound: 0.0000727
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000732, upper bound: 0.0000736
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000732, upper bound: 0.0000735
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000734, upper bound: 0.0000739
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000738, upper bound: 0.0000743
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000734, upper bound: 0.0000736
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000737, upper bound: 0.0000736
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000729, upper bound: 0.0000727
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000729, upper bound: 0.0000727
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000739
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000739
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000731, upper bound: 0.0000733
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000731, upper bound: 0.0000733
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000746, upper bound: 0.0000734
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000751, upper bound: 0.0000736
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000732, upper bound: 0.0000732
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000737, upper bound: 0.0000734
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000741, upper bound: 0.0000737
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000737
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000739, upper bound: 0.0000737
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000737
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000731, upper bound: 0.0000733
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000732, upper bound: 0.0000733
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000739, upper bound: 0.0000739
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000738
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000738, upper bound: 0.0000742
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000738, upper bound: 0.0000742
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000743, upper bound: 0.0000737
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000739
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000739, upper bound: 0.0000734
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000739
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000734, upper bound: 0.0000737
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000732, upper bound: 0.0000732
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000734, upper bound: 0.0000738
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000732, upper bound: 0.0000734
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000735, upper bound: 0.0000736
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000735, upper bound: 0.0000737
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000732
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000738, upper bound: 0.0000732
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000732, upper bound: 0.0000732
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000738, upper bound: 0.0000733
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000737, upper bound: 0.0000742
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000737, upper bound: 0.0000742
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000731, upper bound: 0.0000732
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000732, upper bound: 0.0000732
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000734, upper bound: 0.0000732
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000734, upper bound: 0.0000732
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000741
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000735, upper bound: 0.0000741
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000734, upper bound: 0.0000739
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000734, upper bound: 0.0000734
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000735, upper bound: 0.0000740
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000735, upper bound: 0.0000739
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000727, upper bound: 0.0000728
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000727, upper bound: 0.0000726
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000733, upper bound: 0.0000742
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000733, upper bound: 0.0000737
time: 0.18 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 1.62 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000738
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000739
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000741
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000735, upper bound: 0.0000739
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000735, upper bound: 0.0000741
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000735, upper bound: 0.0000740
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000735, upper bound: 0.0000738
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000735, upper bound: 0.0000735
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000726, upper bound: 0.0000727
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000726, upper bound: 0.0000727
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000732, upper bound: 0.0000736
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000732, upper bound: 0.0000735
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000734, upper bound: 0.0000739
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000738, upper bound: 0.0000743
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000734, upper bound: 0.0000736
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000737, upper bound: 0.0000736
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000729, upper bound: 0.0000727
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000729, upper bound: 0.0000727
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000739
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000739
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000731, upper bound: 0.0000733
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000731, upper bound: 0.0000733
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000746, upper bound: 0.0000734
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000751, upper bound: 0.0000736
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000732, upper bound: 0.0000732
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000737, upper bound: 0.0000734
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000741, upper bound: 0.0000737
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000737
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000739, upper bound: 0.0000737
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000737
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000731, upper bound: 0.0000733
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000732, upper bound: 0.0000733
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000739, upper bound: 0.0000739
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000738
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000738, upper bound: 0.0000742
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000738, upper bound: 0.0000742
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000743, upper bound: 0.0000737
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000739
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000739, upper bound: 0.0000734
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000739
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000734, upper bound: 0.0000737
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000732, upper bound: 0.0000732
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000734, upper bound: 0.0000738
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000732, upper bound: 0.0000734
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000735, upper bound: 0.0000736
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000735, upper bound: 0.0000737
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000732
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000738, upper bound: 0.0000732
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000732, upper bound: 0.0000732
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000738, upper bound: 0.0000733
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000737, upper bound: 0.0000742
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000737, upper bound: 0.0000742
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000731, upper bound: 0.0000732
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000732, upper bound: 0.0000732
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000734, upper bound: 0.0000732
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000734, upper bound: 0.0000732
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000741
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000735, upper bound: 0.0000741
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000734, upper bound: 0.0000739
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000734, upper bound: 0.0000734
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000735, upper bound: 0.0000740
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000735, upper bound: 0.0000739
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000727, upper bound: 0.0000728
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000727, upper bound: 0.0000726
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000733, upper bound: 0.0000742
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.62
Output dim: 0, lower bound: -0.0000733, upper bound: 0.0000737

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000725, upper bound: 0.0000725
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000725, upper bound: 0.0000725
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000734, upper bound: 0.0000731
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000737, upper bound: 0.0000731
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 6

### Candidate
type: DSZ, layer: 3, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000725, upper bound: 0.0000725
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000725, upper bound: 0.0000725
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 6

### Candidate
type: DSZ, layer: 3, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000725, upper bound: 0.0000725
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000725, upper bound: 0.0000725
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 6

### Candidate
type: DSZ, layer: 3, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000725, upper bound: 0.0000725
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000725, upper bound: 0.0000725
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000731, upper bound: 0.0000731
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000736
time: 0.17 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 1.76 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.76
Output dim: 0, lower bound: -0.0000725, upper bound: 0.0000725
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.76
Output dim: 0, lower bound: -0.0000725, upper bound: 0.0000725
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.76
Output dim: 0, lower bound: -0.0000734, upper bound: 0.0000731
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.76
Output dim: 0, lower bound: -0.0000737, upper bound: 0.0000731
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.76
Output dim: 0, lower bound: -0.0000725, upper bound: 0.0000725
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.76
Output dim: 0, lower bound: -0.0000725, upper bound: 0.0000725
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.76
Output dim: 0, lower bound: -0.0000725, upper bound: 0.0000725
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.76
Output dim: 0, lower bound: -0.0000725, upper bound: 0.0000725
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.76
Output dim: 0, lower bound: -0.0000725, upper bound: 0.0000725
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.76
Output dim: 0, lower bound: -0.0000725, upper bound: 0.0000725
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.76
Output dim: 0, lower bound: -0.0000731, upper bound: 0.0000731
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.76
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000736

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 1.58 + 146.72 = 148.29 seconds
