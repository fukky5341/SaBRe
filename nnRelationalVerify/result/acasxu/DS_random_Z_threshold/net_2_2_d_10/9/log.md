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
execution time: IAR + RelationalAnalysis = 0.58 + 1.79 = 2.38 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -198.1763548, upper bound: 198.1763548

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1748416, upper bound: 198.1748416
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1748416, upper bound: 198.1748416
time: 1.11 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.18 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 2.18
Output dim: 0, lower bound: -198.1748416, upper bound: 198.1748416
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 2.18
Output dim: 0, lower bound: -198.1748416, upper bound: 198.1748416

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1748306, upper bound: 198.1748378
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1748306, upper bound: 198.1748306
time: 0.89 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1640589, upper bound: 198.1640589
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1640589, upper bound: 198.1640589
time: 0.66 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.89 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.89
Output dim: 0, lower bound: -198.1748306, upper bound: 198.1748378
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.89
Output dim: 0, lower bound: -198.1748306, upper bound: 198.1748306
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.89
Output dim: 0, lower bound: -198.1640589, upper bound: 198.1640589
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.89
Output dim: 0, lower bound: -198.1640589, upper bound: 198.1640589

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1748088, upper bound: 198.1748108
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1748088, upper bound: 198.1748378
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1725575, upper bound: 198.1725225
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1725215, upper bound: 198.1725575
time: 0.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1640398, upper bound: 198.1640453
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1640398, upper bound: 198.1640398
time: 0.70 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1631545, upper bound: 198.1631545
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1631545, upper bound: 198.1631618
time: 0.58 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.02 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.02
Output dim: 0, lower bound: -198.1748088, upper bound: 198.1748108
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.02
Output dim: 0, lower bound: -198.1748088, upper bound: 198.1748378
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.02
Output dim: 0, lower bound: -198.1725575, upper bound: 198.1725225
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.02
Output dim: 0, lower bound: -198.1725215, upper bound: 198.1725575
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.02
Output dim: 0, lower bound: -198.1640398, upper bound: 198.1640453
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.02
Output dim: 0, lower bound: -198.1640398, upper bound: 198.1640398
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.02
Output dim: 0, lower bound: -198.1631545, upper bound: 198.1631545
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.02
Output dim: 0, lower bound: -198.1631545, upper bound: 198.1631618

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1725575, upper bound: 198.1725230
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1725215, upper bound: 198.1725262
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1746506, upper bound: 198.1746526
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1746510, upper bound: 198.1746526
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1385305, upper bound: 198.1385305
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1385305, upper bound: 198.1385305
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1725208, upper bound: 198.1725208
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1725208, upper bound: 198.1725575
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1640172, upper bound: 198.1640172
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1640172, upper bound: 198.1640173
time: 0.97 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1631386, upper bound: 198.1631386
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1631386, upper bound: 198.1631386
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1631395, upper bound: 198.1631395
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1631395, upper bound: 198.1631395
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1631386, upper bound: 198.1631475
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1631386, upper bound: 198.1631386
time: 0.68 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.88 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 0, lower bound: -198.1725575, upper bound: 198.1725230
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 0, lower bound: -198.1725215, upper bound: 198.1725262
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 0, lower bound: -198.1746506, upper bound: 198.1746526
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 0, lower bound: -198.1746510, upper bound: 198.1746526
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 0, lower bound: -198.1385305, upper bound: 198.1385305
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 0, lower bound: -198.1385305, upper bound: 198.1385305
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 0, lower bound: -198.1725208, upper bound: 198.1725208
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 0, lower bound: -198.1725208, upper bound: 198.1725575
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 0, lower bound: -198.1640172, upper bound: 198.1640172
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 0, lower bound: -198.1640172, upper bound: 198.1640173
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 0, lower bound: -198.1631386, upper bound: 198.1631386
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 0, lower bound: -198.1631386, upper bound: 198.1631386
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 0, lower bound: -198.1631395, upper bound: 198.1631395
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 0, lower bound: -198.1631395, upper bound: 198.1631395
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 0, lower bound: -198.1631386, upper bound: 198.1631475
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.88
Output dim: 0, lower bound: -198.1631386, upper bound: 198.1631386

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1725115, upper bound: 198.1725151
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1725115, upper bound: 198.1725148
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1385305, upper bound: 198.1385305
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1385305, upper bound: 198.1385305
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1746253, upper bound: 198.1746303
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1746253, upper bound: 198.1746273
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1389238, upper bound: 198.1389238
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1389238, upper bound: 198.1389238
time: 1.08 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1385305, upper bound: 198.1385305
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1385305, upper bound: 198.1385305
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1385067, upper bound: 198.1385067
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1385067, upper bound: 198.1385067
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1370334, upper bound: 198.1370334
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1370334, upper bound: 198.1370334
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1724945, upper bound: 198.1725100
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1724945, upper bound: 198.1725260
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 17

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1631236, upper bound: 198.1631236
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1631236, upper bound: 198.1631236
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 17

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1631236, upper bound: 198.1631236
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1631236, upper bound: 198.1631236
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1631236, upper bound: 198.1631236
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1631236, upper bound: 198.1631236
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 26

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 17

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1631236, upper bound: 198.1631236
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1631236, upper bound: 198.1631236
time: 0.92 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1631236, upper bound: 198.1631236
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1631236, upper bound: 198.1631236
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1631236, upper bound: 198.1631236
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1631236, upper bound: 198.1631236
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1631236, upper bound: 198.1631236
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1631236, upper bound: 198.1631236
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1631236, upper bound: 198.1631236
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1631236, upper bound: 198.1631236
time: 0.59 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 1.94 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 0, lower bound: -198.1725115, upper bound: 198.1725151
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 0, lower bound: -198.1725115, upper bound: 198.1725148
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 0, lower bound: -198.1385305, upper bound: 198.1385305
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 0, lower bound: -198.1385305, upper bound: 198.1385305
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 0, lower bound: -198.1746253, upper bound: 198.1746303
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 0, lower bound: -198.1746253, upper bound: 198.1746273
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 0, lower bound: -198.1389238, upper bound: 198.1389238
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 0, lower bound: -198.1389238, upper bound: 198.1389238
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 0, lower bound: -198.1385305, upper bound: 198.1385305
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 0, lower bound: -198.1385305, upper bound: 198.1385305
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 0, lower bound: -198.1385067, upper bound: 198.1385067
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 0, lower bound: -198.1385067, upper bound: 198.1385067
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 0, lower bound: -198.1370334, upper bound: 198.1370334
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 0, lower bound: -198.1370334, upper bound: 198.1370334
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 0, lower bound: -198.1724945, upper bound: 198.1725100
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 0, lower bound: -198.1724945, upper bound: 198.1725260
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 0, lower bound: -198.1631236, upper bound: 198.1631236
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 0, lower bound: -198.1631236, upper bound: 198.1631236
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 0, lower bound: -198.1631236, upper bound: 198.1631236
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 0, lower bound: -198.1631236, upper bound: 198.1631236
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 0, lower bound: -198.1631236, upper bound: 198.1631236
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 0, lower bound: -198.1631236, upper bound: 198.1631236
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 0, lower bound: -198.1631236, upper bound: 198.1631236
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 0, lower bound: -198.1631236, upper bound: 198.1631236
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 0, lower bound: -198.1631236, upper bound: 198.1631236
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 0, lower bound: -198.1631236, upper bound: 198.1631236
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 0, lower bound: -198.1631236, upper bound: 198.1631236
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 0, lower bound: -198.1631236, upper bound: 198.1631236
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 0, lower bound: -198.1631236, upper bound: 198.1631236
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 0, lower bound: -198.1631236, upper bound: 198.1631236
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 0, lower bound: -198.1631236, upper bound: 198.1631236
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 0, lower bound: -198.1631236, upper bound: 198.1631236

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1725103, upper bound: 198.1725128
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1725115, upper bound: 198.1725103
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1717004, upper bound: 198.1716989
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1716989, upper bound: 198.1717018
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1385279, upper bound: 198.1385279
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1385279, upper bound: 198.1385279
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1385305, upper bound: 198.1385305
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1385305, upper bound: 198.1385305
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1724891, upper bound: 198.1724905
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1724891, upper bound: 198.1724937
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1746253, upper bound: 198.1746253
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1746253, upper bound: 198.1746273
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1374314, upper bound: 198.1374314
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1374314, upper bound: 198.1374314
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1386886, upper bound: 198.1386886
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1386886, upper bound: 198.1386886
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1370334, upper bound: 198.1370334
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1370334, upper bound: 198.1370334
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1370334, upper bound: 198.1370334
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1370334, upper bound: 198.1370334
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1382683, upper bound: 198.1382683
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1382683, upper bound: 198.1382683
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1370036, upper bound: 198.1370036
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1370036, upper bound: 198.1370036
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1370334, upper bound: 198.1370334
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1370334, upper bound: 198.1370334
time: 0.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1370036, upper bound: 198.1370036
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1370036, upper bound: 198.1370036
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1724945, upper bound: 198.1724981
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1724945, upper bound: 198.1725100
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1716769, upper bound: 198.1716787
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1716769, upper bound: 198.1717040
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 26

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 17

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1631227, upper bound: 198.1631052
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1631052, upper bound: 198.1631227
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 26

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 17

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1631236, upper bound: 198.1631128
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1631128, upper bound: 198.1631236
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 26

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1631227, upper bound: 198.1631052
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1631052, upper bound: 198.1631227
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 26

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1625829, upper bound: 198.1625829
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1625829, upper bound: 198.1625829
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 26

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 43

Time for candidate selection: 0.58 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1612078, upper bound: 198.1612084
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1612084, upper bound: 198.1612071
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 26

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 17

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 12

Time for candidate selection: 0.60 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1612071, upper bound: 198.1612084
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1612084, upper bound: 198.1612071
time: 0.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 26

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 17

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 5

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1627541, upper bound: 198.1627519
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1627519, upper bound: 198.1627541
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 17

### Candidate
type: DSZ, layer: 1, pos: 26

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 35

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1630719, upper bound: 198.1631036
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1631036, upper bound: 198.1630719
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.83 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1630719, upper bound: 198.1631036
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1631036, upper bound: 198.1630719
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 17

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.85 seconds

### Candidate
type: DSZ, layer: 3, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1625829, upper bound: 198.1625829
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1625829, upper bound: 198.1625829
time: 0.87 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 26

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1587130, upper bound: 198.1587130
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1587130, upper bound: 198.1587130
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 26

### Candidate
type: DSZ, layer: 1, pos: 17

### Candidate
type: DSZ, layer: 1, pos: 19

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1627541, upper bound: 198.1627519
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1627519, upper bound: 198.1627541
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 26

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 32

Time for candidate selection: 1.01 seconds

### Candidate
type: DSZ, layer: 3, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1625829, upper bound: 198.1625829
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1625829, upper bound: 198.1625829
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 26

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 32

Time for candidate selection: 1.04 seconds

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1630979, upper bound: 198.1631236
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1631236, upper bound: 198.1631017
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 26

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 5

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 43

Time for candidate selection: 1.02 seconds

### Candidate
type: DSZ, layer: 3, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1617839, upper bound: 198.1617848
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1617848, upper bound: 198.1617839
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 37

Time for candidate selection: 1.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1587130, upper bound: 198.1587130
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1587130, upper bound: 198.1587130
time: 0.53 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.72 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1725103, upper bound: 198.1725128
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1725115, upper bound: 198.1725103
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1717004, upper bound: 198.1716989
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1716989, upper bound: 198.1717018
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1385279, upper bound: 198.1385279
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1385279, upper bound: 198.1385279
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1385305, upper bound: 198.1385305
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1385305, upper bound: 198.1385305
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1724891, upper bound: 198.1724905
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1724891, upper bound: 198.1724937
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1746253, upper bound: 198.1746253
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1746253, upper bound: 198.1746273
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1374314, upper bound: 198.1374314
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1374314, upper bound: 198.1374314
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1386886, upper bound: 198.1386886
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1386886, upper bound: 198.1386886
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1370334, upper bound: 198.1370334
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1370334, upper bound: 198.1370334
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1370334, upper bound: 198.1370334
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1370334, upper bound: 198.1370334
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1382683, upper bound: 198.1382683
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1382683, upper bound: 198.1382683
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1370036, upper bound: 198.1370036
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1370036, upper bound: 198.1370036
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1370334, upper bound: 198.1370334
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1370334, upper bound: 198.1370334
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1370036, upper bound: 198.1370036
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1370036, upper bound: 198.1370036
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1724945, upper bound: 198.1724981
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1724945, upper bound: 198.1725100
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1716769, upper bound: 198.1716787
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1716769, upper bound: 198.1717040
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1631227, upper bound: 198.1631052
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1631052, upper bound: 198.1631227
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1631236, upper bound: 198.1631128
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1631128, upper bound: 198.1631236
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1631227, upper bound: 198.1631052
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1631052, upper bound: 198.1631227
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1625829, upper bound: 198.1625829
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1625829, upper bound: 198.1625829
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1612078, upper bound: 198.1612084
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1612084, upper bound: 198.1612071
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1612071, upper bound: 198.1612084
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1612084, upper bound: 198.1612071
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1627541, upper bound: 198.1627519
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1627519, upper bound: 198.1627541
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1630719, upper bound: 198.1631036
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1631036, upper bound: 198.1630719
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1630719, upper bound: 198.1631036
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1631036, upper bound: 198.1630719
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1625829, upper bound: 198.1625829
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1625829, upper bound: 198.1625829
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1587130, upper bound: 198.1587130
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1587130, upper bound: 198.1587130
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1627541, upper bound: 198.1627519
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1627519, upper bound: 198.1627541
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1625829, upper bound: 198.1625829
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1625829, upper bound: 198.1625829
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1630979, upper bound: 198.1631236
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1631236, upper bound: 198.1631017
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1617839, upper bound: 198.1617848
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1617848, upper bound: 198.1617839
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1587130, upper bound: 198.1587130
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.72
Output dim: 0, lower bound: -198.1587130, upper bound: 198.1587130

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1385279, upper bound: 198.1385279
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1385279, upper bound: 198.1385279
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1385279, upper bound: 198.1385279
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1385279, upper bound: 198.1385279
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1716989, upper bound: 198.1716989
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1716989, upper bound: 198.1716989
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1382914, upper bound: 198.1382914
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1382914, upper bound: 198.1382914
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1385032, upper bound: 198.1385032
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1385032, upper bound: 198.1385032
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1370263, upper bound: 198.1370263
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1370263, upper bound: 198.1370263
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1382914, upper bound: 198.1382914
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1382914, upper bound: 198.1382914
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1370334, upper bound: 198.1370334
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1370334, upper bound: 198.1370334
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1724891, upper bound: 198.1724891
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1724891, upper bound: 198.1724905
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1385067, upper bound: 198.1385067
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1385067, upper bound: 198.1385067
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1739353, upper bound: 198.1739353
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1739353, upper bound: 198.1739353
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1745802, upper bound: 198.1745802
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1745802, upper bound: 198.1745814
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1374014, upper bound: 198.1374014
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1374014, upper bound: 198.1374014
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1370334, upper bound: 198.1370334
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1370334, upper bound: 198.1370334
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1386639, upper bound: 198.1386639
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1386639, upper bound: 198.1386639
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1386639, upper bound: 198.1386639
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1386639, upper bound: 198.1386639
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1370036, upper bound: 198.1370036
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1370036, upper bound: 198.1370036
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1370334, upper bound: 198.1370334
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1370334, upper bound: 198.1370334
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1382683, upper bound: 198.1382683
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1382683, upper bound: 198.1382683
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1382683, upper bound: 198.1382683
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1382683, upper bound: 198.1382683
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367688, upper bound: 198.1367688
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367688, upper bound: 198.1367688
time: 0.91 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367688, upper bound: 198.1367688
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367688, upper bound: 198.1367688
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1370334, upper bound: 198.1370334
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1370334, upper bound: 198.1370334
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1369968, upper bound: 198.1369968
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1369968, upper bound: 198.1369968
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1370036, upper bound: 198.1370036
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1370036, upper bound: 198.1370036
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1370036, upper bound: 198.1370036
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1370036, upper bound: 198.1370036
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1724046, upper bound: 198.1724046
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1724046, upper bound: 198.1724100
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1716769, upper bound: 198.1716787
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1716769, upper bound: 198.1716769
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1716769, upper bound: 198.1716807
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1716769, upper bound: 198.1717040
time: 0.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1611853, upper bound: 198.1611492
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1611865, upper bound: 198.1611660
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1630912, upper bound: 198.1631227
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1631058, upper bound: 198.1630970
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1630981, upper bound: 198.1631128
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1631236, upper bound: 198.1630899
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1625610, upper bound: 198.1625631
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1625637, upper bound: 198.1625610
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1586846, upper bound: 198.1586806
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1586846, upper bound: 198.1586806
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1586806, upper bound: 198.1586846
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1586806, upper bound: 198.1586846
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1617613, upper bound: 198.1617613
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1617613, upper bound: 198.1617613
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1625558, upper bound: 198.1625558
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1625558, upper bound: 198.1625558
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1609047, upper bound: 198.1609190
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1609047, upper bound: 198.1609017
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1595471, upper bound: 198.1595459
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1595471, upper bound: 198.1595454
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1609047, upper bound: 198.1609190
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1609178, upper bound: 198.1609017
time: 0.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1595461, upper bound: 198.1595459
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1595471, upper bound: 198.1595454
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1627507, upper bound: 198.1627411
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1627424, upper bound: 198.1627485
time: 0.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1586379, upper bound: 198.1586444
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1586379, upper bound: 198.1586444
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 12

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1630719, upper bound: 198.1630926
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1630713, upper bound: 198.1631036
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1625411, upper bound: 198.1625182
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1625429, upper bound: 198.1625225
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1611521, upper bound: 198.1611883
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1611546, upper bound: 198.1611871
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1630713, upper bound: 198.1630511
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1630848, upper bound: 198.1630446
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1613810, upper bound: 198.1613810
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1613810, upper bound: 198.1613810
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1601748, upper bound: 198.1601748
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1601748, upper bound: 198.1601748
time: 0.87 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1

### Candidate
type: DSZ, layer: 3, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1587127, upper bound: 198.1587130
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1587130, upper bound: 198.1587127
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Candidate
type: DSZ, layer: 3, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1577551, upper bound: 198.1577558
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1577558, upper bound: 198.1577551
time: 0.94 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1609047, upper bound: 198.1609190
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1609017, upper bound: 198.1609178
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1618787, upper bound: 198.1618838
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1618787, upper bound: 198.1618932
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1625792, upper bound: 198.1625792
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1625792, upper bound: 198.1625792
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1601748, upper bound: 198.1601748
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1601748, upper bound: 198.1601748
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1630970, upper bound: 198.1631071
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1630883, upper bound: 198.1631227
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1625829, upper bound: 198.1625829
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1625829, upper bound: 198.1625829
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1617811, upper bound: 198.1617598
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1617616, upper bound: 198.1617821
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1595459, upper bound: 198.1595461
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1595471, upper bound: 198.1595454
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1584683, upper bound: 198.1584677
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1584677, upper bound: 198.1584683
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 34
type: DSZ, layer: 3, pos: 24
type: DSZ, layer: 3, pos: 47
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 29
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 21
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1586882, upper bound: 198.1587130
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1587130, upper bound: 198.1586882
time: 1.34 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 2.88 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1385279, upper bound: 198.1385279
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1385279, upper bound: 198.1385279
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1385279, upper bound: 198.1385279
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1385279, upper bound: 198.1385279
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1716989, upper bound: 198.1716989
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1716989, upper bound: 198.1716989
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1382914, upper bound: 198.1382914
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1382914, upper bound: 198.1382914
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1385032, upper bound: 198.1385032
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1385032, upper bound: 198.1385032
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1370263, upper bound: 198.1370263
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1370263, upper bound: 198.1370263
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1382914, upper bound: 198.1382914
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1382914, upper bound: 198.1382914
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1370334, upper bound: 198.1370334
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1370334, upper bound: 198.1370334
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1724891, upper bound: 198.1724891
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1724891, upper bound: 198.1724905
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1385067, upper bound: 198.1385067
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1385067, upper bound: 198.1385067
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1739353, upper bound: 198.1739353
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1739353, upper bound: 198.1739353
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1745802, upper bound: 198.1745802
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1745802, upper bound: 198.1745814
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1374014, upper bound: 198.1374014
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1374014, upper bound: 198.1374014
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1370334, upper bound: 198.1370334
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1370334, upper bound: 198.1370334
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1386639, upper bound: 198.1386639
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1386639, upper bound: 198.1386639
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1386639, upper bound: 198.1386639
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1386639, upper bound: 198.1386639
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1370036, upper bound: 198.1370036
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1370036, upper bound: 198.1370036
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1370334, upper bound: 198.1370334
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1370334, upper bound: 198.1370334
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1382683, upper bound: 198.1382683
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1382683, upper bound: 198.1382683
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1382683, upper bound: 198.1382683
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1382683, upper bound: 198.1382683
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1367688, upper bound: 198.1367688
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1367688, upper bound: 198.1367688
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1367688, upper bound: 198.1367688
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1367688, upper bound: 198.1367688
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1370334, upper bound: 198.1370334
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1370334, upper bound: 198.1370334
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1369968, upper bound: 198.1369968
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1369968, upper bound: 198.1369968
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1370036, upper bound: 198.1370036
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1370036, upper bound: 198.1370036
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1370036, upper bound: 198.1370036
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1370036, upper bound: 198.1370036
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1724046, upper bound: 198.1724046
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1724046, upper bound: 198.1724100
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1716769, upper bound: 198.1716787
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1716769, upper bound: 198.1716769
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1716769, upper bound: 198.1716807
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1716769, upper bound: 198.1717040
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1611853, upper bound: 198.1611492
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1611865, upper bound: 198.1611660
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1630912, upper bound: 198.1631227
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1631058, upper bound: 198.1630970
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1630981, upper bound: 198.1631128
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1631236, upper bound: 198.1630899
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1625610, upper bound: 198.1625631
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1625637, upper bound: 198.1625610
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1586846, upper bound: 198.1586806
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1586846, upper bound: 198.1586806
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1586806, upper bound: 198.1586846
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1586806, upper bound: 198.1586846
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1617613, upper bound: 198.1617613
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1617613, upper bound: 198.1617613
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1625558, upper bound: 198.1625558
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1625558, upper bound: 198.1625558
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1609047, upper bound: 198.1609190
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1609047, upper bound: 198.1609017
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1595471, upper bound: 198.1595459
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1595471, upper bound: 198.1595454
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1609047, upper bound: 198.1609190
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1609178, upper bound: 198.1609017
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1595461, upper bound: 198.1595459
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1595471, upper bound: 198.1595454
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1627507, upper bound: 198.1627411
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1627424, upper bound: 198.1627485
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1586379, upper bound: 198.1586444
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1586379, upper bound: 198.1586444
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1630719, upper bound: 198.1630926
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1630713, upper bound: 198.1631036
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1625411, upper bound: 198.1625182
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1625429, upper bound: 198.1625225
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1611521, upper bound: 198.1611883
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1611546, upper bound: 198.1611871
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1630713, upper bound: 198.1630511
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1630848, upper bound: 198.1630446
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1613810, upper bound: 198.1613810
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1613810, upper bound: 198.1613810
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1601748, upper bound: 198.1601748
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1601748, upper bound: 198.1601748
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1587127, upper bound: 198.1587130
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1587130, upper bound: 198.1587127
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1577551, upper bound: 198.1577558
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1577558, upper bound: 198.1577551
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1609047, upper bound: 198.1609190
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1609017, upper bound: 198.1609178
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1618787, upper bound: 198.1618838
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1618787, upper bound: 198.1618932
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1625792, upper bound: 198.1625792
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1625792, upper bound: 198.1625792
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1601748, upper bound: 198.1601748
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1601748, upper bound: 198.1601748
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1630970, upper bound: 198.1631071
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1630883, upper bound: 198.1631227
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1625829, upper bound: 198.1625829
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1625829, upper bound: 198.1625829
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1617811, upper bound: 198.1617598
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1617616, upper bound: 198.1617821
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1595459, upper bound: 198.1595461
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1595471, upper bound: 198.1595454
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1584683, upper bound: 198.1584677
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1584677, upper bound: 198.1584683
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1586882, upper bound: 198.1587130
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.88
Output dim: 0, lower bound: -198.1587130, upper bound: 198.1586882

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1385032, upper bound: 198.1385032
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1385032, upper bound: 198.1385032
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1382890, upper bound: 198.1382890
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1382890, upper bound: 198.1382890
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1382890, upper bound: 198.1382890
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1382890, upper bound: 198.1382890
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1382890, upper bound: 198.1382890
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1382890, upper bound: 198.1382890
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1716989, upper bound: 198.1716989
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1716989, upper bound: 198.1716989
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1686474, upper bound: 198.1686474
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1686474, upper bound: 198.1686474
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1382890, upper bound: 198.1382890
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1382890, upper bound: 198.1382890
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1369968, upper bound: 198.1369968
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1369968, upper bound: 198.1369968
time: 0.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1385032, upper bound: 198.1385032
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1385032, upper bound: 198.1385032
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1370263, upper bound: 198.1370263
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1370263, upper bound: 198.1370263
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1369968, upper bound: 198.1369968
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1369968, upper bound: 198.1369968
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1382683, upper bound: 198.1382683
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1382683, upper bound: 198.1382683
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1724891, upper bound: 198.1724891
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1724891, upper bound: 198.1724891
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1716766, upper bound: 198.1716766
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1716766, upper bound: 198.1716784
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1370036, upper bound: 198.1370036
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1370036, upper bound: 198.1370036
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1370036, upper bound: 198.1370036
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1370036, upper bound: 198.1370036
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1709682, upper bound: 198.1709682
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1709682, upper bound: 198.1709682
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371665, upper bound: 198.1371665
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371665, upper bound: 198.1371665
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1724037, upper bound: 198.1724037
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1724037, upper bound: 198.1724037
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1724037, upper bound: 198.1724045
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1724037, upper bound: 198.1724046
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1370036, upper bound: 198.1370036
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1370036, upper bound: 198.1370036
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371665, upper bound: 198.1371665
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371665, upper bound: 198.1371665
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1370263, upper bound: 198.1370263
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1370263, upper bound: 198.1370263
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1370036, upper bound: 198.1370036
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1370036, upper bound: 198.1370036
time: 0.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1382683, upper bound: 198.1382683
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1382683, upper bound: 198.1382683
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371665, upper bound: 198.1371665
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371665, upper bound: 198.1371665
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371665, upper bound: 198.1371665
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1371665, upper bound: 198.1371665
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1386603, upper bound: 198.1386603
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1386603, upper bound: 198.1386603
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367920, upper bound: 198.1367920
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367920, upper bound: 198.1367920
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367920, upper bound: 198.1367920
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367920, upper bound: 198.1367920
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1369968, upper bound: 198.1369968
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1369968, upper bound: 198.1369968
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367688, upper bound: 198.1367688
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367688, upper bound: 198.1367688
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1370263, upper bound: 198.1370263
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1370263, upper bound: 198.1370263
time: 0.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367920, upper bound: 198.1367920
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367920, upper bound: 198.1367920
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367920, upper bound: 198.1367920
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367920, upper bound: 198.1367920
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1382683, upper bound: 198.1382683
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1382683, upper bound: 198.1382683
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1382683, upper bound: 198.1382683
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1382683, upper bound: 198.1382683
time: 1.07 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1382648, upper bound: 198.1382648
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1382648, upper bound: 198.1382648
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1382648, upper bound: 198.1382648
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1382648, upper bound: 198.1382648
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367688, upper bound: 198.1367688
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367688, upper bound: 198.1367688
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367688, upper bound: 198.1367688
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367688, upper bound: 198.1367688
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367688, upper bound: 198.1367688
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367688, upper bound: 198.1367688
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367623, upper bound: 198.1367623
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367623, upper bound: 198.1367623
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1370263, upper bound: 198.1370263
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1370263, upper bound: 198.1370263
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367990, upper bound: 198.1367990
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367623, upper bound: 198.1367623
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367623, upper bound: 198.1367623
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367623, upper bound: 198.1367623
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367623, upper bound: 198.1367623
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1370036, upper bound: 198.1370036
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1370036, upper bound: 198.1370036
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367688, upper bound: 198.1367688
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367688, upper bound: 198.1367688
time: 1.08 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367688, upper bound: 198.1367688
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1367688, upper bound: 198.1367688
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1369968, upper bound: 198.1369968
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1369968, upper bound: 198.1369968
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831
1: -67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032
2: -58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826
3: -92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407
4: -72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751

Time for backsubstitution: 0.83 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 2.38 + 418.31 = 420.68 seconds
