## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_2.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 9)
Time budget: 420 seconds
Split limit: 100
Threshold: 198.13671952904002


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831)
1: (-67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032)
2: (-58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826)
3: (-92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407)
4: (-72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.41 + 1.88 = 3.28 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -198.1763548, upper bound: 198.1763548

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1393660, upper bound: 198.1393660
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1393660, upper bound: 198.1393660
time: 0.87 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.88 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.88
Output dim: 0, lower bound: -198.1393660, upper bound: 198.1393660
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.88
Output dim: 0, lower bound: -198.1393660, upper bound: 198.1393660

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1393660, upper bound: 198.1393660
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1393660, upper bound: 198.1393660
time: 0.87 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1393660, upper bound: 198.1393660
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1393660, upper bound: 198.1393660
time: 0.87 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.77 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.77
Output dim: 0, lower bound: -198.1393660, upper bound: 198.1393660
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.77
Output dim: 0, lower bound: -198.1393660, upper bound: 198.1393660
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.77
Output dim: 0, lower bound: -198.1393660, upper bound: 198.1393660
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.77
Output dim: 0, lower bound: -198.1393660, upper bound: 198.1393660

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1378750, upper bound: 198.1378750
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1378750, upper bound: 198.1378750
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1378750, upper bound: 198.1378750
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1378750, upper bound: 198.1378750
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1378750, upper bound: 198.1378750
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1378750, upper bound: 198.1378750
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1378750, upper bound: 198.1378750
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1378750, upper bound: 198.1378750
time: 0.59 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.69 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.69
Output dim: 0, lower bound: -198.1378750, upper bound: 198.1378750
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.69
Output dim: 0, lower bound: -198.1378750, upper bound: 198.1378750
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.69
Output dim: 0, lower bound: -198.1378750, upper bound: 198.1378750
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.69
Output dim: 0, lower bound: -198.1378750, upper bound: 198.1378750
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.69
Output dim: 0, lower bound: -198.1378750, upper bound: 198.1378750
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.69
Output dim: 0, lower bound: -198.1378750, upper bound: 198.1378750
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.69
Output dim: 0, lower bound: -198.1378750, upper bound: 198.1378750
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.69
Output dim: 0, lower bound: -198.1378750, upper bound: 198.1378750

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1374314, upper bound: 198.1374314
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1374314, upper bound: 198.1374314
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1374314, upper bound: 198.1374314
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1374314, upper bound: 198.1374314
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1374314, upper bound: 198.1374314
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1374314, upper bound: 198.1374314
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1374314, upper bound: 198.1374314
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1374314, upper bound: 198.1374314
time: 0.91 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1374314, upper bound: 198.1374314
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1374314, upper bound: 198.1374314
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1374314, upper bound: 198.1374314
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1374314, upper bound: 198.1374314
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1374314, upper bound: 198.1374314
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1374314, upper bound: 198.1374314
time: 0.85 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1374314, upper bound: 198.1374314
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1374314, upper bound: 198.1374314
time: 0.90 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.89 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 0, lower bound: -198.1374314, upper bound: 198.1374314
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 0, lower bound: -198.1374314, upper bound: 198.1374314
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 0, lower bound: -198.1374314, upper bound: 198.1374314
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 0, lower bound: -198.1374314, upper bound: 198.1374314
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 0, lower bound: -198.1374314, upper bound: 198.1374314
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 0, lower bound: -198.1374314, upper bound: 198.1374314
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 0, lower bound: -198.1374314, upper bound: 198.1374314
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 0, lower bound: -198.1374314, upper bound: 198.1374314
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 0, lower bound: -198.1374314, upper bound: 198.1374314
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 0, lower bound: -198.1374314, upper bound: 198.1374314
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 0, lower bound: -198.1374314, upper bound: 198.1374314
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 0, lower bound: -198.1374314, upper bound: 198.1374314
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 0, lower bound: -198.1374314, upper bound: 198.1374314
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 0, lower bound: -198.1374314, upper bound: 198.1374314
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 0, lower bound: -198.1374314, upper bound: 198.1374314
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 0, lower bound: -198.1374314, upper bound: 198.1374314

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.49 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 3.04 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.90 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.90 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.70 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
time: 0.52 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.66 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.66
Output dim: 0, lower bound: -198.1371970, upper bound: 198.1371970

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 1.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.94 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.86 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.88 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.87 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.96 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.86 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.87 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.86 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.80 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 3.21 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367688, upper bound: 198.1367688
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367688, upper bound: 198.1367688
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367688, upper bound: 198.1367688
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367688, upper bound: 198.1367688
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367688, upper bound: 198.1367688
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367688, upper bound: 198.1367688
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367688, upper bound: 198.1367688
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367688, upper bound: 198.1367688
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367688, upper bound: 198.1367688
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367688, upper bound: 198.1367688
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367688, upper bound: 198.1367688
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367688, upper bound: 198.1367688
time: 0.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367688, upper bound: 198.1367688
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367688, upper bound: 198.1367688
time: 0.64 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 3.27 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.27
Output dim: 0, lower bound: -198.1367688, upper bound: 198.1367688
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.27
Output dim: 0, lower bound: -198.1367688, upper bound: 198.1367688
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.27
Output dim: 0, lower bound: -198.1367688, upper bound: 198.1367688
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.27
Output dim: 0, lower bound: -198.1367688, upper bound: 198.1367688
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.27
Output dim: 0, lower bound: -198.1367688, upper bound: 198.1367688
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.27
Output dim: 0, lower bound: -198.1367688, upper bound: 198.1367688
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.27
Output dim: 0, lower bound: -198.1367688, upper bound: 198.1367688
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.27
Output dim: 0, lower bound: -198.1367688, upper bound: 198.1367688
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.27
Output dim: 0, lower bound: -198.1367688, upper bound: 198.1367688
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.27
Output dim: 0, lower bound: -198.1367688, upper bound: 198.1367688
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.27
Output dim: 0, lower bound: -198.1367688, upper bound: 198.1367688
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.27
Output dim: 0, lower bound: -198.1367688, upper bound: 198.1367688
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.27
Output dim: 0, lower bound: -198.1367688, upper bound: 198.1367688
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.27
Output dim: 0, lower bound: -198.1367688, upper bound: 198.1367688
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 3.28 + 418.22 = 421.50 seconds
