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
execution time: IAR + RelationalAnalysis = 1.11 + 0.53 = 1.64 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0000773, upper bound: 0.0000773

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000772, upper bound: 0.0000772
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000771, upper bound: 0.0000772
time: 0.17 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.42 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.42
Output dim: 0, lower bound: -0.0000772, upper bound: 0.0000772
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.42
Output dim: 0, lower bound: -0.0000771, upper bound: 0.0000772

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000764, upper bound: 0.0000761
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000766, upper bound: 0.0000754
time: 0.14 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000753, upper bound: 0.0000766
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000761, upper bound: 0.0000764
time: 0.16 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.42 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.42
Output dim: 0, lower bound: -0.0000764, upper bound: 0.0000761
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.42
Output dim: 0, lower bound: -0.0000766, upper bound: 0.0000754
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.42
Output dim: 0, lower bound: -0.0000753, upper bound: 0.0000766
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.42
Output dim: 0, lower bound: -0.0000761, upper bound: 0.0000764

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000762, upper bound: 0.0000761
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000761
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000765, upper bound: 0.0000752
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000763, upper bound: 0.0000754
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000753, upper bound: 0.0000763
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000752, upper bound: 0.0000765
time: 0.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000761, upper bound: 0.0000755
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000761, upper bound: 0.0000762
time: 0.15 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 1.41 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.41
Output dim: 0, lower bound: -0.0000762, upper bound: 0.0000761
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.41
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000761
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.41
Output dim: 0, lower bound: -0.0000765, upper bound: 0.0000752
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.41
Output dim: 0, lower bound: -0.0000763, upper bound: 0.0000754
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.41
Output dim: 0, lower bound: -0.0000753, upper bound: 0.0000763
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.41
Output dim: 0, lower bound: -0.0000752, upper bound: 0.0000765
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.41
Output dim: 0, lower bound: -0.0000761, upper bound: 0.0000755
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.41
Output dim: 0, lower bound: -0.0000761, upper bound: 0.0000762

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000758, upper bound: 0.0000755
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000756
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000748, upper bound: 0.0000752
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000751, upper bound: 0.0000755
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000760, upper bound: 0.0000746
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000759, upper bound: 0.0000746
time: 0.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000759, upper bound: 0.0000746
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000759, upper bound: 0.0000747
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000759
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000746, upper bound: 0.0000759
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000746, upper bound: 0.0000759
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000746, upper bound: 0.0000760
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000756, upper bound: 0.0000751
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000752, upper bound: 0.0000748
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000755
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000758
time: 0.15 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 1.44 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.44
Output dim: 0, lower bound: -0.0000758, upper bound: 0.0000755
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.44
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000756
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.44
Output dim: 0, lower bound: -0.0000748, upper bound: 0.0000752
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.44
Output dim: 0, lower bound: -0.0000751, upper bound: 0.0000755
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.44
Output dim: 0, lower bound: -0.0000760, upper bound: 0.0000746
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.44
Output dim: 0, lower bound: -0.0000759, upper bound: 0.0000746
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.44
Output dim: 0, lower bound: -0.0000759, upper bound: 0.0000746
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.44
Output dim: 0, lower bound: -0.0000759, upper bound: 0.0000747
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.44
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000759
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.44
Output dim: 0, lower bound: -0.0000746, upper bound: 0.0000759
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.44
Output dim: 0, lower bound: -0.0000746, upper bound: 0.0000759
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.44
Output dim: 0, lower bound: -0.0000746, upper bound: 0.0000760
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.44
Output dim: 0, lower bound: -0.0000756, upper bound: 0.0000751
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.44
Output dim: 0, lower bound: -0.0000752, upper bound: 0.0000748
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.44
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000755
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.44
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000758

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000755
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000757, upper bound: 0.0000748
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000755
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000754, upper bound: 0.0000748
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000745, upper bound: 0.0000751
time: 0.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000746, upper bound: 0.0000747
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000746, upper bound: 0.0000755
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000749, upper bound: 0.0000748
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000744
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000760, upper bound: 0.0000744
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000744
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000758, upper bound: 0.0000744
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000746, upper bound: 0.0000744
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000758, upper bound: 0.0000745
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000747
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000758, upper bound: 0.0000746
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000758
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000747
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000745, upper bound: 0.0000758
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000744, upper bound: 0.0000746
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000744, upper bound: 0.0000758
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000744, upper bound: 0.0000747
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000744, upper bound: 0.0000760
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000744, upper bound: 0.0000748
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000749
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000754, upper bound: 0.0000746
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000746
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000751, upper bound: 0.0000745
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000754
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000747
time: 0.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.09 seconds

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
- Time for DS candidates: 1.49 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.49
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000755
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.49
Output dim: 0, lower bound: -0.0000757, upper bound: 0.0000748
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.49
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000755
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.49
Output dim: 0, lower bound: -0.0000754, upper bound: 0.0000748
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.49
Output dim: 0, lower bound: -0.0000745, upper bound: 0.0000751
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.49
Output dim: 0, lower bound: -0.0000746, upper bound: 0.0000747
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.49
Output dim: 0, lower bound: -0.0000746, upper bound: 0.0000755
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.49
Output dim: 0, lower bound: -0.0000749, upper bound: 0.0000748
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.49
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000744
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.49
Output dim: 0, lower bound: -0.0000760, upper bound: 0.0000744
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.49
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000744
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.49
Output dim: 0, lower bound: -0.0000758, upper bound: 0.0000744
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.49
Output dim: 0, lower bound: -0.0000746, upper bound: 0.0000744
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.49
Output dim: 0, lower bound: -0.0000758, upper bound: 0.0000745
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.49
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000747
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.49
Output dim: 0, lower bound: -0.0000758, upper bound: 0.0000746
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.49
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000758
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.49
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000747
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.49
Output dim: 0, lower bound: -0.0000745, upper bound: 0.0000758
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.49
Output dim: 0, lower bound: -0.0000744, upper bound: 0.0000746
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.49
Output dim: 0, lower bound: -0.0000744, upper bound: 0.0000758
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.49
Output dim: 0, lower bound: -0.0000744, upper bound: 0.0000747
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.49
Output dim: 0, lower bound: -0.0000744, upper bound: 0.0000760
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.49
Output dim: 0, lower bound: -0.0000744, upper bound: 0.0000748
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.49
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000749
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.49
Output dim: 0, lower bound: -0.0000754, upper bound: 0.0000746
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.49
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000746
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.49
Output dim: 0, lower bound: -0.0000751, upper bound: 0.0000745
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.49
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000754
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.49
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000747
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.49
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000757
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.49
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000748

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000743
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000741
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000738, upper bound: 0.0000740
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000741, upper bound: 0.0000740
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000743
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000742
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000738, upper bound: 0.0000736
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000741, upper bound: 0.0000741
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000737, upper bound: 0.0000742
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000741
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000737, upper bound: 0.0000740
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000738, upper bound: 0.0000740
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000738, upper bound: 0.0000743
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000738, upper bound: 0.0000742
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000738, upper bound: 0.0000741
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000739, upper bound: 0.0000741
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000737
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000737
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000736
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000737
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000737
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000737
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000736
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000737
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000738, upper bound: 0.0000736
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000736
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000738, upper bound: 0.0000736
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000738
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000739
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000738
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000739
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000739
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000739, upper bound: 0.0000742
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000739, upper bound: 0.0000740
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000738, upper bound: 0.0000740
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000739, upper bound: 0.0000740
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000738, upper bound: 0.0000742
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000738
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000736
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000738
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000737, upper bound: 0.0000742
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000740
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000737, upper bound: 0.0000740
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000737, upper bound: 0.0000740
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000737, upper bound: 0.0000742
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000740
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000737, upper bound: 0.0000741
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000737, upper bound: 0.0000740
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000739
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000738
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000738
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000743, upper bound: 0.0000738
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000738
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000737
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000741, upper bound: 0.0000736
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000737
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000741, upper bound: 0.0000741
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000738
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000740
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000743, upper bound: 0.0000740
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000741
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000738
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000741, upper bound: 0.0000740
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000743, upper bound: 0.0000740
time: 0.17 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 1.67 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000743
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000741
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000738, upper bound: 0.0000740
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000741, upper bound: 0.0000740
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000743
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000742
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000738, upper bound: 0.0000736
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000741, upper bound: 0.0000741
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000737, upper bound: 0.0000742
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000741
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000737, upper bound: 0.0000740
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000738, upper bound: 0.0000740
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000738, upper bound: 0.0000743
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000738, upper bound: 0.0000742
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000738, upper bound: 0.0000741
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000739, upper bound: 0.0000741
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000737
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000737
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000736
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000737
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000737
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000737
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000736
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000737
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000738, upper bound: 0.0000736
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000736
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000738, upper bound: 0.0000736
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000738
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000739
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000738
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000739
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000739
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000739, upper bound: 0.0000742
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000739, upper bound: 0.0000740
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000738, upper bound: 0.0000740
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000739, upper bound: 0.0000740
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000738, upper bound: 0.0000742
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000738
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000736
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000738
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000737, upper bound: 0.0000742
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000740
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000737, upper bound: 0.0000740
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000737, upper bound: 0.0000740
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000737, upper bound: 0.0000742
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000740
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000737, upper bound: 0.0000741
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000737, upper bound: 0.0000740
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000739
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000738
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000738
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000743, upper bound: 0.0000738
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000738
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000737
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000741, upper bound: 0.0000736
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000737
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000741, upper bound: 0.0000741
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000738
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000740
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000743, upper bound: 0.0000740
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000741
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000738
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000741, upper bound: 0.0000740
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.67
Output dim: 0, lower bound: -0.0000743, upper bound: 0.0000740

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000739, upper bound: 0.0000741
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000738, upper bound: 0.0000741
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

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
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000738, upper bound: 0.0000742
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000738, upper bound: 0.0000742
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000737, upper bound: 0.0000742
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000737, upper bound: 0.0000742
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000741, upper bound: 0.0000735
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000741, upper bound: 0.0000736
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000741
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000735, upper bound: 0.0000741
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000737
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000737
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000738
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000738
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284
1: -0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442
2: -0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118
3: -0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389
4: -0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000741, upper bound: 0.0000738
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000741, upper bound: 0.0000739
time: 0.19 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 1.58 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.58
Output dim: 0, lower bound: -0.0000739, upper bound: 0.0000741
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.58
Output dim: 0, lower bound: -0.0000738, upper bound: 0.0000741
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.58
Output dim: 0, lower bound: -0.0000738, upper bound: 0.0000742
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.58
Output dim: 0, lower bound: -0.0000738, upper bound: 0.0000742
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.58
Output dim: 0, lower bound: -0.0000737, upper bound: 0.0000742
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.58
Output dim: 0, lower bound: -0.0000737, upper bound: 0.0000742
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.58
Output dim: 0, lower bound: -0.0000741, upper bound: 0.0000735
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.58
Output dim: 0, lower bound: -0.0000741, upper bound: 0.0000736
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.58
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000741
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.58
Output dim: 0, lower bound: -0.0000735, upper bound: 0.0000741
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.58
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000737
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.58
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000737
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.58
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000738
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.58
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000738
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.58
Output dim: 0, lower bound: -0.0000741, upper bound: 0.0000738
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.58
Output dim: 0, lower bound: -0.0000741, upper bound: 0.0000739

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 1.64 + 107.84 = 109.49 seconds
