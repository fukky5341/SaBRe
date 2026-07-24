## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_8.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 2)
Time budget: 420 seconds
Split limit: 100
Threshold: 0.019857408


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372)
1: (-0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658)
2: (0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212)
3: (-0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573)
4: (0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.98 + 0.61 = 1.60 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0206848, upper bound: 0.0206848

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0203767, upper bound: 0.0203767
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0203767, upper bound: 0.0203767
time: 0.16 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.34 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.34
Output dim: 0, lower bound: -0.0203767, upper bound: 0.0203767
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.34
Output dim: 0, lower bound: -0.0203767, upper bound: 0.0203767

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0203570, upper bound: 0.0203572
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0203572, upper bound: 0.0203570
time: 0.15 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0202263, upper bound: 0.0202793
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0202793, upper bound: 0.0202263
time: 0.16 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.45 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.45
Output dim: 0, lower bound: -0.0203570, upper bound: 0.0203572
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.45
Output dim: 0, lower bound: -0.0203572, upper bound: 0.0203570
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.45
Output dim: 0, lower bound: -0.0202263, upper bound: 0.0202793
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.45
Output dim: 0, lower bound: -0.0202793, upper bound: 0.0202263

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0202045, upper bound: 0.0202765
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0202309, upper bound: 0.0202066
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0202066, upper bound: 0.0202309
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0202765, upper bound: 0.0202045
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0202045, upper bound: 0.0202765
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0202066, upper bound: 0.0202309
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0202309, upper bound: 0.0202066
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0202765, upper bound: 0.0202045
time: 0.16 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 1.44 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.44
Output dim: 0, lower bound: -0.0202045, upper bound: 0.0202765
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.44
Output dim: 0, lower bound: -0.0202309, upper bound: 0.0202066
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.44
Output dim: 0, lower bound: -0.0202066, upper bound: 0.0202309
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.44
Output dim: 0, lower bound: -0.0202765, upper bound: 0.0202045
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.44
Output dim: 0, lower bound: -0.0202045, upper bound: 0.0202765
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.44
Output dim: 0, lower bound: -0.0202066, upper bound: 0.0202309
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.44
Output dim: 0, lower bound: -0.0202309, upper bound: 0.0202066
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.44
Output dim: 0, lower bound: -0.0202765, upper bound: 0.0202045

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0202045, upper bound: 0.0202599
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0201826, upper bound: 0.0202765
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0194561, upper bound: 0.0194236
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0194417, upper bound: 0.0194433
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 3, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0201819, upper bound: 0.0202024
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0201816, upper bound: 0.0202001
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0202416, upper bound: 0.0201812
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0202460, upper bound: 0.0201814
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 33

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199211, upper bound: 0.0199835
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199312, upper bound: 0.0199170
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 33

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0201819, upper bound: 0.0202024
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0201816, upper bound: 0.0202001
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 33

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0202001, upper bound: 0.0201816
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0202024, upper bound: 0.0201819
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 33

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0195032, upper bound: 0.0194237
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0194822, upper bound: 0.0194412
time: 0.16 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 1.49 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.49
Output dim: 0, lower bound: -0.0202045, upper bound: 0.0202599
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.49
Output dim: 0, lower bound: -0.0201826, upper bound: 0.0202765
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 1.49
Output dim: 0, lower bound: -0.0194561, upper bound: 0.0194236
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 1.49
Output dim: 0, lower bound: -0.0194417, upper bound: 0.0194433
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.49
Output dim: 0, lower bound: -0.0201819, upper bound: 0.0202024
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.49
Output dim: 0, lower bound: -0.0201816, upper bound: 0.0202001
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.49
Output dim: 0, lower bound: -0.0202416, upper bound: 0.0201812
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.49
Output dim: 0, lower bound: -0.0202460, upper bound: 0.0201814
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.49
Output dim: 0, lower bound: -0.0199211, upper bound: 0.0199835
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.49
Output dim: 0, lower bound: -0.0199312, upper bound: 0.0199170
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.49
Output dim: 0, lower bound: -0.0201819, upper bound: 0.0202024
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.49
Output dim: 0, lower bound: -0.0201816, upper bound: 0.0202001
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.49
Output dim: 0, lower bound: -0.0202001, upper bound: 0.0201816
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.49
Output dim: 0, lower bound: -0.0202024, upper bound: 0.0201819
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 1.49
Output dim: 0, lower bound: -0.0195032, upper bound: 0.0194237
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 1.49
Output dim: 0, lower bound: -0.0194822, upper bound: 0.0194412

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0202045, upper bound: 0.0201835
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0201930, upper bound: 0.0202467
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0201628, upper bound: 0.0202460
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0201628, upper bound: 0.0202416
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199140, upper bound: 0.0199454
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199305, upper bound: 0.0199180
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0201816, upper bound: 0.0201730
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0201629, upper bound: 0.0202001
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0194956, upper bound: 0.0194126
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0194729, upper bound: 0.0194255
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0201507, upper bound: 0.0201708
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0202353, upper bound: 0.0201534
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191213, upper bound: 0.0191810
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191196, upper bound: 0.0191845
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199248, upper bound: 0.0199156
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199290, upper bound: 0.0199140
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 8

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0194411, upper bound: 0.0194370
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0194206, upper bound: 0.0194514
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0201793, upper bound: 0.0201807
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0201794, upper bound: 0.0201958
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0201958, upper bound: 0.0201794
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0201807, upper bound: 0.0201793
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0201507, upper bound: 0.0201714
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0201917, upper bound: 0.0201551
time: 0.17 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 2.02 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -0.0202045, upper bound: 0.0201835
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -0.0201930, upper bound: 0.0202467
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -0.0201628, upper bound: 0.0202460
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -0.0201628, upper bound: 0.0202416
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -0.0199140, upper bound: 0.0199454
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -0.0199305, upper bound: 0.0199180
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -0.0201816, upper bound: 0.0201730
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -0.0201629, upper bound: 0.0202001
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.02
Output dim: 0, lower bound: -0.0194956, upper bound: 0.0194126
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.02
Output dim: 0, lower bound: -0.0194729, upper bound: 0.0194255
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -0.0201507, upper bound: 0.0201708
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -0.0202353, upper bound: 0.0201534
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.02
Output dim: 0, lower bound: -0.0191213, upper bound: 0.0191810
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.02
Output dim: 0, lower bound: -0.0191196, upper bound: 0.0191845
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -0.0199248, upper bound: 0.0199156
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -0.0199290, upper bound: 0.0199140
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.02
Output dim: 0, lower bound: -0.0194411, upper bound: 0.0194370
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.02
Output dim: 0, lower bound: -0.0194206, upper bound: 0.0194514
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -0.0201793, upper bound: 0.0201807
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -0.0201794, upper bound: 0.0201958
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -0.0201958, upper bound: 0.0201794
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -0.0201807, upper bound: 0.0201793
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -0.0201507, upper bound: 0.0201714
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -0.0201917, upper bound: 0.0201551

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0194412, upper bound: 0.0193951
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0194237, upper bound: 0.0194398
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199149, upper bound: 0.0199570
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199151, upper bound: 0.0199147
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0194263, upper bound: 0.0194747
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0193950, upper bound: 0.0194949
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199143, upper bound: 0.0199603
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199148, upper bound: 0.0199140
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199140, upper bound: 0.0199317
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199140, upper bound: 0.0199454
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199122
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199247, upper bound: 0.0199081
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0201793, upper bound: 0.0201666
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0201794, upper bound: 0.0201687
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0201596, upper bound: 0.0201807
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0201612, upper bound: 0.0201958
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199191
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199081
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0202353, upper bound: 0.0201507
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0202233, upper bound: 0.0201534
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199221, upper bound: 0.0199141
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199140, upper bound: 0.0199156
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199290, upper bound: 0.0199140
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199147, upper bound: 0.0199140
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199152, upper bound: 0.0199149
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199283, upper bound: 0.0199140
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 29

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 8

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0201794, upper bound: 0.0201687
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0201612, upper bound: 0.0201958
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 29

### Candidate
type: DSZ, layer: 3, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0201958, upper bound: 0.0201612
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0201687, upper bound: 0.0201794
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 29

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0201490, upper bound: 0.0201686
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0201702, upper bound: 0.0201490
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 8

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0201507, upper bound: 0.0201550
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0201507, upper bound: 0.0201714
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199122, upper bound: 0.0199081
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199395, upper bound: 0.0199081
time: 0.17 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 1.34 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0194412, upper bound: 0.0193951
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0194237, upper bound: 0.0194398
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0199149, upper bound: 0.0199570
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0199151, upper bound: 0.0199147
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0194263, upper bound: 0.0194747
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0193950, upper bound: 0.0194949
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0199143, upper bound: 0.0199603
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0199148, upper bound: 0.0199140
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0199140, upper bound: 0.0199317
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0199140, upper bound: 0.0199454
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199122
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0199247, upper bound: 0.0199081
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0201793, upper bound: 0.0201666
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0201794, upper bound: 0.0201687
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0201596, upper bound: 0.0201807
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0201612, upper bound: 0.0201958
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199191
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199081
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0202353, upper bound: 0.0201507
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0202233, upper bound: 0.0201534
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0199221, upper bound: 0.0199141
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0199140, upper bound: 0.0199156
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0199290, upper bound: 0.0199140
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0199147, upper bound: 0.0199140
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0199152, upper bound: 0.0199149
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0199283, upper bound: 0.0199140
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0201794, upper bound: 0.0201687
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0201612, upper bound: 0.0201958
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0201958, upper bound: 0.0201612
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0201687, upper bound: 0.0201794
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0201490, upper bound: 0.0201686
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0201702, upper bound: 0.0201490
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0201507, upper bound: 0.0201550
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0201507, upper bound: 0.0201714
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0199122, upper bound: 0.0199081
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.34
Output dim: 0, lower bound: -0.0199395, upper bound: 0.0199081

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199088, upper bound: 0.0199511
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199089, upper bound: 0.0199086
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199145, upper bound: 0.0199142
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199145, upper bound: 0.0199140
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 29

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199083, upper bound: 0.0199546
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199084, upper bound: 0.0199081
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199083, upper bound: 0.0199081
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199089, upper bound: 0.0199081
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 29

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199140, upper bound: 0.0199146
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199140, upper bound: 0.0199273
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 29

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191175, upper bound: 0.0191440
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191175, upper bound: 0.0191462
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199084
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199122
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199233, upper bound: 0.0199081
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199083, upper bound: 0.0199081
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199140, upper bound: 0.0199140
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199295, upper bound: 0.0199140
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 25

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199140, upper bound: 0.0199199
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199145, upper bound: 0.0199140
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 29

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0201490, upper bound: 0.0201702
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0201495, upper bound: 0.0201490
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 29

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0201490, upper bound: 0.0201852
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0201510, upper bound: 0.0201490
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199084
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199191
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199081
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199081
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0202267, upper bound: 0.0201490
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0201763, upper bound: 0.0201490
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199083, upper bound: 0.0199081
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199711, upper bound: 0.0199081
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199083
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199164, upper bound: 0.0199081
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 8

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191231, upper bound: 0.0191183
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191175, upper bound: 0.0191187
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199276, upper bound: 0.0199140
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199145, upper bound: 0.0199140
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191232, upper bound: 0.0191175
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191184, upper bound: 0.0191175
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 5

### Candidate
type: DSZ, layer: 3, pos: 8

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199091
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199094, upper bound: 0.0199081
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199081
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199225, upper bound: 0.0199081
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 25

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0201490, upper bound: 0.0201591
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0201687, upper bound: 0.0201490
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 25

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0194137, upper bound: 0.0194313
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0193913, upper bound: 0.0194471
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 25

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0194471, upper bound: 0.0193913
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0194313, upper bound: 0.0194137
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 29

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199140, upper bound: 0.0199145
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199199, upper bound: 0.0199140
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0201490, upper bound: 0.0201495
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0201490, upper bound: 0.0201686
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199081
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199081
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0193822, upper bound: 0.0193891
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0193823, upper bound: 0.0194207
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0193822, upper bound: 0.0194099
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0193823, upper bound: 0.0194301
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199122, upper bound: 0.0199081
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199084, upper bound: 0.0199081
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 25

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 8

### Candidate
type: DSZ, layer: 3, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199353, upper bound: 0.0199081
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199089, upper bound: 0.0199081
time: 0.20 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 1.72 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0199088, upper bound: 0.0199511
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0199089, upper bound: 0.0199086
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0199145, upper bound: 0.0199142
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0199145, upper bound: 0.0199140
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0199083, upper bound: 0.0199546
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0199084, upper bound: 0.0199081
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0199083, upper bound: 0.0199081
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0199089, upper bound: 0.0199081
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0199140, upper bound: 0.0199146
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0199140, upper bound: 0.0199273
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0191175, upper bound: 0.0191440
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0191175, upper bound: 0.0191462
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199084
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199122
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0199233, upper bound: 0.0199081
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0199083, upper bound: 0.0199081
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0199140, upper bound: 0.0199140
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0199295, upper bound: 0.0199140
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0199140, upper bound: 0.0199199
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0199145, upper bound: 0.0199140
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0201490, upper bound: 0.0201702
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0201495, upper bound: 0.0201490
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0201490, upper bound: 0.0201852
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0201510, upper bound: 0.0201490
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199084
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199191
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199081
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199081
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0202267, upper bound: 0.0201490
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0201763, upper bound: 0.0201490
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0199083, upper bound: 0.0199081
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0199711, upper bound: 0.0199081
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199083
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0199164, upper bound: 0.0199081
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0191231, upper bound: 0.0191183
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0191175, upper bound: 0.0191187
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0199276, upper bound: 0.0199140
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0199145, upper bound: 0.0199140
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0191232, upper bound: 0.0191175
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0191184, upper bound: 0.0191175
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199091
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0199094, upper bound: 0.0199081
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199081
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0199225, upper bound: 0.0199081
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0201490, upper bound: 0.0201591
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0201687, upper bound: 0.0201490
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0194137, upper bound: 0.0194313
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0193913, upper bound: 0.0194471
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0194471, upper bound: 0.0193913
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0194313, upper bound: 0.0194137
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0199140, upper bound: 0.0199145
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0199199, upper bound: 0.0199140
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0201490, upper bound: 0.0201495
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0201490, upper bound: 0.0201686
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199081
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199081
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0193822, upper bound: 0.0193891
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0193823, upper bound: 0.0194207
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0193822, upper bound: 0.0194099
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0193823, upper bound: 0.0194301
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0199122, upper bound: 0.0199081
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0199084, upper bound: 0.0199081
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0199353, upper bound: 0.0199081
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.72
Output dim: 0, lower bound: -0.0199089, upper bound: 0.0199081

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199449
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199083, upper bound: 0.0199341
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 8

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191129, upper bound: 0.0191090
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191087
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191425, upper bound: 0.0191182
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191175, upper bound: 0.0191187
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199083, upper bound: 0.0199081
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199087, upper bound: 0.0199081
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191100, upper bound: 0.0191531
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191093, upper bound: 0.0191570
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191108, upper bound: 0.0191086
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191093, upper bound: 0.0191086
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 8

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 29

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191101, upper bound: 0.0191086
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191093, upper bound: 0.0191086
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 8

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191139, upper bound: 0.0191086
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191095, upper bound: 0.0191086
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199087
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199081
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 29

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 25

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199215
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199081
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 29

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199084
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199084
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 25

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199083
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199122
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 8

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191195, upper bound: 0.0191086
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191172, upper bound: 0.0191086
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 25

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191169, upper bound: 0.0191088
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191087
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191183, upper bound: 0.0191175
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191181, upper bound: 0.0191285
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 25

### Candidate
type: DSZ, layer: 3, pos: 5

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191267, upper bound: 0.0191175
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191262, upper bound: 0.0191175
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191183, upper bound: 0.0191248
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191175, upper bound: 0.0191310
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199081
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199087, upper bound: 0.0199081
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0193813, upper bound: 0.0193805
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0193809, upper bound: 0.0194213
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0193824, upper bound: 0.0193805
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0193810, upper bound: 0.0193805
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199215
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199081
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199081
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199087, upper bound: 0.0199081
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 25

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 29

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199081
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199084
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 8

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199081
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199164
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199081
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199081
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 29

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191087, upper bound: 0.0191086
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191089, upper bound: 0.0191103
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 29

### Candidate
type: DSZ, layer: 3, pos: 25

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0194752, upper bound: 0.0193805
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0194561, upper bound: 0.0193949
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 29

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0194265, upper bound: 0.0193808
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0193871, upper bound: 0.0193869
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 29

### Candidate
type: DSZ, layer: 3, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191100, upper bound: 0.0191088
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191093, upper bound: 0.0191094
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191730, upper bound: 0.0191086
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191684, upper bound: 0.0191094
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 5

### Candidate
type: DSZ, layer: 3, pos: 29

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191094, upper bound: 0.0191086
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191088, upper bound: 0.0191100
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 5

### Candidate
type: DSZ, layer: 3, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 8

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191119, upper bound: 0.0191086
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191099, upper bound: 0.0191086
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199083, upper bound: 0.0199081
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199217, upper bound: 0.0199081
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 8

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 29

### Candidate
type: DSZ, layer: 3, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191232, upper bound: 0.0191175
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191175, upper bound: 0.0191175
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 25

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191092, upper bound: 0.0191086
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191092, upper bound: 0.0191419
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191097, upper bound: 0.0191086
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191094, upper bound: 0.0191086
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 25

### Candidate
type: DSZ, layer: 3, pos: 5

### Candidate
type: DSZ, layer: 3, pos: 8

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191093, upper bound: 0.0191086
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191092, upper bound: 0.0191117
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 29

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199225, upper bound: 0.0199081
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199089, upper bound: 0.0199081
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 8

### Candidate
type: DSZ, layer: 3, pos: 29

### Candidate
type: DSZ, layer: 3, pos: 25

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199262
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199081
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199082, upper bound: 0.0199081
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199087, upper bound: 0.0199081
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199087
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199081
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 29

### Candidate
type: DSZ, layer: 3, pos: 25

### Candidate
type: DSZ, layer: 3, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199081
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199142, upper bound: 0.0199081
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 5

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199089
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199081
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199236
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199081
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191092
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191093
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199081
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199081
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191118, upper bound: 0.0191086
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191114, upper bound: 0.0191091
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 25

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191118, upper bound: 0.0191089
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191091
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199353, upper bound: 0.0199081
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199215, upper bound: 0.0199081
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191242, upper bound: 0.0191086
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191086
time: 0.20 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 1.53 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199449
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0199083, upper bound: 0.0199341
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0191129, upper bound: 0.0191090
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191087
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0191425, upper bound: 0.0191182
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0191175, upper bound: 0.0191187
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0199083, upper bound: 0.0199081
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0199087, upper bound: 0.0199081
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0191100, upper bound: 0.0191531
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0191093, upper bound: 0.0191570
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0191108, upper bound: 0.0191086
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0191093, upper bound: 0.0191086
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0191101, upper bound: 0.0191086
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0191093, upper bound: 0.0191086
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0191139, upper bound: 0.0191086
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0191095, upper bound: 0.0191086
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199087
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199081
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199215
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199081
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199084
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199084
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199083
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199122
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0191195, upper bound: 0.0191086
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0191172, upper bound: 0.0191086
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0191169, upper bound: 0.0191088
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191087
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0191183, upper bound: 0.0191175
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0191181, upper bound: 0.0191285
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0191267, upper bound: 0.0191175
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0191262, upper bound: 0.0191175
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0191183, upper bound: 0.0191248
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0191175, upper bound: 0.0191310
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199081
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0199087, upper bound: 0.0199081
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0193813, upper bound: 0.0193805
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0193809, upper bound: 0.0194213
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0193824, upper bound: 0.0193805
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0193810, upper bound: 0.0193805
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199215
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199081
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199081
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0199087, upper bound: 0.0199081
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199081
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199084
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199081
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199164
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199081
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199081
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0191087, upper bound: 0.0191086
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0191089, upper bound: 0.0191103
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0194752, upper bound: 0.0193805
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0194561, upper bound: 0.0193949
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0194265, upper bound: 0.0193808
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0193871, upper bound: 0.0193869
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0191100, upper bound: 0.0191088
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0191093, upper bound: 0.0191094
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0191730, upper bound: 0.0191086
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0191684, upper bound: 0.0191094
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0191094, upper bound: 0.0191086
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0191088, upper bound: 0.0191100
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0191119, upper bound: 0.0191086
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0191099, upper bound: 0.0191086
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0199083, upper bound: 0.0199081
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0199217, upper bound: 0.0199081
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0191232, upper bound: 0.0191175
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0191175, upper bound: 0.0191175
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0191092, upper bound: 0.0191086
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0191092, upper bound: 0.0191419
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0191097, upper bound: 0.0191086
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0191094, upper bound: 0.0191086
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0191093, upper bound: 0.0191086
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0191092, upper bound: 0.0191117
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0199225, upper bound: 0.0199081
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0199089, upper bound: 0.0199081
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199262
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199081
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0199082, upper bound: 0.0199081
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0199087, upper bound: 0.0199081
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199087
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199081
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199081
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0199142, upper bound: 0.0199081
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199089
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199081
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199236
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199081
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191092
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191093
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199081
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0199081, upper bound: 0.0199081
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0191118, upper bound: 0.0191086
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0191114, upper bound: 0.0191091
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0191118, upper bound: 0.0191089
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191091
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0199353, upper bound: 0.0199081
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0199215, upper bound: 0.0199081
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0191242, upper bound: 0.0191086
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191086

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 8

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191109, upper bound: 0.0191440
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191469
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.79 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191100, upper bound: 0.0191333
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191375
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.53 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 5

### Candidate
type: DSZ, layer: 3, pos: 29

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 8

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191101, upper bound: 0.0191086
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191086
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 29

### Candidate
type: DSZ, layer: 3, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 5

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191150, upper bound: 0.0191086
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191086
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191086
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191221
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 5

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 29

### Candidate
type: DSZ, layer: 3, pos: 25

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191086
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191086
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 29

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 25

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 5

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191234
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191265
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 25

### Candidate
type: DSZ, layer: 3, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 5

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191088
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191087
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 25

### Candidate
type: DSZ, layer: 3, pos: 5

### Candidate
type: DSZ, layer: 3, pos: 29

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191091, upper bound: 0.0191086
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191089, upper bound: 0.0191105
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 25

### Candidate
type: DSZ, layer: 3, pos: 29

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191091, upper bound: 0.0191086
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191088, upper bound: 0.0191118
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 5

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191090, upper bound: 0.0191093
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191098
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191091, upper bound: 0.0191114
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191118
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191092, upper bound: 0.0191086
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191086
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 5

### Candidate
type: DSZ, layer: 3, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 25

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191148, upper bound: 0.0191086
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191086
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 29

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 8

### Candidate
type: DSZ, layer: 3, pos: 5

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191091, upper bound: 0.0191221
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191254
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 29

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 8

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 5

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191093, upper bound: 0.0191086
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191086
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 8

### Candidate
type: DSZ, layer: 3, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 5

### Candidate
type: DSZ, layer: 3, pos: 29

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191094, upper bound: 0.0191086
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191086
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 8

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 29

### Candidate
type: DSZ, layer: 3, pos: 5

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191137, upper bound: 0.0191086
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191086
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191087, upper bound: 0.0191086
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191088, upper bound: 0.0191133
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191088
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191104
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191087, upper bound: 0.0191086
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191088, upper bound: 0.0191133
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 8

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191099
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191119
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 8

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 29

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191087, upper bound: 0.0191086
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191088, upper bound: 0.0191105
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 29

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 8

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191086
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191086
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191097, upper bound: 0.0191086
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191094, upper bound: 0.0191086
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 5

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 25

### Candidate
type: DSZ, layer: 3, pos: 8

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 29

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191133, upper bound: 0.0191086
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191128, upper bound: 0.0191086
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 5

### Candidate
type: DSZ, layer: 3, pos: 25

### Candidate
type: DSZ, layer: 3, pos: 29

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191134, upper bound: 0.0191086
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191131, upper bound: 0.0191086
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 5

### Candidate
type: DSZ, layer: 3, pos: 8

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 25

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191104, upper bound: 0.0191086
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191095, upper bound: 0.0191086
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 8

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191092, upper bound: 0.0191243
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191464
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191092, upper bound: 0.0191087
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191093
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 25

### Candidate
type: DSZ, layer: 3, pos: 5

### Candidate
type: DSZ, layer: 3, pos: 8

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 29

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191097, upper bound: 0.0191088
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191087
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 8

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 5

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 25

### Candidate
type: DSZ, layer: 3, pos: 29

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191136, upper bound: 0.0191086
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191086
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 5

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 25

### Candidate
type: DSZ, layer: 3, pos: 29

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 8

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191086
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191148
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 8

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 5

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191086
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191092
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 29

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 5

### Candidate
type: DSZ, layer: 3, pos: 25

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191086
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191094
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 29

### Candidate
type: DSZ, layer: 3, pos: 19

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191212, upper bound: 0.0191086
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191161, upper bound: 0.0191090
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191095
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191107
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 29

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191091
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191094
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191171
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191169
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 29

### Candidate
type: DSZ, layer: 3, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191092
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191094
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Candidate
type: DSZ, layer: 3, pos: 8

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191197, upper bound: 0.0191090
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191091
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191183, upper bound: 0.0191091
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191091
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 5

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191325, upper bound: 0.0191086
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191309, upper bound: 0.0191086
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372
1: -0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658
2: 0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212
3: -0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573
4: 0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 5
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 19
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 39

### Candidate
type: DSZ, layer: 3, pos: 31

### Candidate
type: DSZ, layer: 3, pos: 5

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191265, upper bound: 0.0191086
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0191234, upper bound: 0.0191086
time: 0.20 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 1.60 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191109, upper bound: 0.0191440
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191469
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191100, upper bound: 0.0191333
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191375
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191101, upper bound: 0.0191086
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191086
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191150, upper bound: 0.0191086
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191086
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191086
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191221
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191086
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191086
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191234
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191265
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191088
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191087
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191091, upper bound: 0.0191086
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191089, upper bound: 0.0191105
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191091, upper bound: 0.0191086
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191088, upper bound: 0.0191118
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191090, upper bound: 0.0191093
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191098
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191091, upper bound: 0.0191114
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191118
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191092, upper bound: 0.0191086
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191086
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191148, upper bound: 0.0191086
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191086
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191091, upper bound: 0.0191221
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191254
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191093, upper bound: 0.0191086
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191086
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191094, upper bound: 0.0191086
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191086
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191137, upper bound: 0.0191086
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191086
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191087, upper bound: 0.0191086
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191088, upper bound: 0.0191133
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191088
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191104
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191087, upper bound: 0.0191086
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191088, upper bound: 0.0191133
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191099
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191119
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191087, upper bound: 0.0191086
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191088, upper bound: 0.0191105
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191086
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191086
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191097, upper bound: 0.0191086
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191094, upper bound: 0.0191086
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191133, upper bound: 0.0191086
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191128, upper bound: 0.0191086
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191134, upper bound: 0.0191086
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191131, upper bound: 0.0191086
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191104, upper bound: 0.0191086
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191095, upper bound: 0.0191086
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191092, upper bound: 0.0191243
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191464
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191092, upper bound: 0.0191087
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191093
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191097, upper bound: 0.0191088
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191087
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191136, upper bound: 0.0191086
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191086
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191086
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191148
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191086
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191092
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191086
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191094
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191212, upper bound: 0.0191086
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191161, upper bound: 0.0191090
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191095
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191107
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191091
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191094
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191171
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191169
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191092
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191094
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191197, upper bound: 0.0191090
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191091
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191183, upper bound: 0.0191091
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191086, upper bound: 0.0191091
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191325, upper bound: 0.0191086
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191309, upper bound: 0.0191086
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191265, upper bound: 0.0191086
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.60
Output dim: 0, lower bound: -0.0191234, upper bound: 0.0191086

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 1.60 + 278.30 = 279.89 seconds
