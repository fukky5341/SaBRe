## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_8.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 7)
Time budget: 420 seconds
Split limit: 100
Threshold: 4810.657341545514


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062)
1: (-294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011)
2: (-466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004)
3: (-542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918)
4: (-407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.74 + 1.90 = 4.65 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -4810.7054486, upper bound: 4810.7054486

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.7020311, upper bound: 4810.7011834
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.7011834, upper bound: 4810.7020311
time: 0.66 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.57 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.57
Output dim: 0, lower bound: -4810.7020311, upper bound: 4810.7011834
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.57
Output dim: 0, lower bound: -4810.7011834, upper bound: 4810.7020311

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6911489, upper bound: 4810.6905460
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6911489, upper bound: 4810.6905460
time: 0.68 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6905460, upper bound: 4810.6911489
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6905460, upper bound: 4810.6911489
time: 0.65 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 4.08 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.08
Output dim: 0, lower bound: -4810.6911489, upper bound: 4810.6905460
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.08
Output dim: 0, lower bound: -4810.6911489, upper bound: 4810.6905460
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.08
Output dim: 0, lower bound: -4810.6905460, upper bound: 4810.6911489
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.08
Output dim: 0, lower bound: -4810.6905460, upper bound: 4810.6911489

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6903633, upper bound: 4810.6899859
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6905731, upper bound: 4810.6895684
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6903633, upper bound: 4810.6899859
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6905731, upper bound: 4810.6895684
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6895684, upper bound: 4810.6905731
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6899859, upper bound: 4810.6903633
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6895684, upper bound: 4810.6905731
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6899859, upper bound: 4810.6903633
time: 0.73 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 4.16 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.16
Output dim: 0, lower bound: -4810.6903633, upper bound: 4810.6899859
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.16
Output dim: 0, lower bound: -4810.6905731, upper bound: 4810.6895684
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.16
Output dim: 0, lower bound: -4810.6903633, upper bound: 4810.6899859
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.16
Output dim: 0, lower bound: -4810.6905731, upper bound: 4810.6895684
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.16
Output dim: 0, lower bound: -4810.6895684, upper bound: 4810.6905731
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.16
Output dim: 0, lower bound: -4810.6899859, upper bound: 4810.6903633
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.16
Output dim: 0, lower bound: -4810.6895684, upper bound: 4810.6905731
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.16
Output dim: 0, lower bound: -4810.6899859, upper bound: 4810.6903633

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6903389, upper bound: 4810.6890127
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6898218, upper bound: 4810.6899538
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6905492, upper bound: 4810.6892914
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6891985, upper bound: 4810.6895125
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6903389, upper bound: 4810.6890127
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6898218, upper bound: 4810.6899538
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6905492, upper bound: 4810.6892914
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6891985, upper bound: 4810.6895125
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6895125, upper bound: 4810.6891985
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6892914, upper bound: 4810.6905492
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6899538, upper bound: 4810.6898218
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6903389
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6895125, upper bound: 4810.6891985
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6892914, upper bound: 4810.6905492
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6899538, upper bound: 4810.6898218
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6903389
time: 0.62 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 3.98 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 0, lower bound: -4810.6903389, upper bound: 4810.6890127
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 0, lower bound: -4810.6898218, upper bound: 4810.6899538
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 0, lower bound: -4810.6905492, upper bound: 4810.6892914
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 0, lower bound: -4810.6891985, upper bound: 4810.6895125
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 0, lower bound: -4810.6903389, upper bound: 4810.6890127
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 0, lower bound: -4810.6898218, upper bound: 4810.6899538
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 0, lower bound: -4810.6905492, upper bound: 4810.6892914
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 0, lower bound: -4810.6891985, upper bound: 4810.6895125
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 0, lower bound: -4810.6895125, upper bound: 4810.6891985
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 0, lower bound: -4810.6892914, upper bound: 4810.6905492
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 0, lower bound: -4810.6899538, upper bound: 4810.6898218
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6903389
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 0, lower bound: -4810.6895125, upper bound: 4810.6891985
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 0, lower bound: -4810.6892914, upper bound: 4810.6905492
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 0, lower bound: -4810.6899538, upper bound: 4810.6898218
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6903389

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6903389, upper bound: 4810.6890127
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6899036, upper bound: 4810.6890127
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6898218, upper bound: 4810.6899538
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6905492, upper bound: 4810.6892914
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6892349, upper bound: 4810.6890127
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890217, upper bound: 4810.6893146
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6891985, upper bound: 4810.6895125
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6903389, upper bound: 4810.6890127
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6899036, upper bound: 4810.6890127
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6898218, upper bound: 4810.6899538
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6905492, upper bound: 4810.6892914
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6892349, upper bound: 4810.6890127
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890217, upper bound: 4810.6893146
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6891985, upper bound: 4810.6895125
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6895125, upper bound: 4810.6891985
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6893146, upper bound: 4810.6890217
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6892349
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6892914, upper bound: 4810.6905492
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6899538, upper bound: 4810.6898218
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6899036
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6903389
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6895125, upper bound: 4810.6891985
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6893146, upper bound: 4810.6890217
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6892349
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6892914, upper bound: 4810.6905492
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6899538, upper bound: 4810.6898218
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6899036
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6903389
time: 0.60 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 4.08 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -4810.6903389, upper bound: 4810.6890127
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -4810.6899036, upper bound: 4810.6890127
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -4810.6898218, upper bound: 4810.6899538
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -4810.6905492, upper bound: 4810.6892914
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -4810.6892349, upper bound: 4810.6890127
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -4810.6890217, upper bound: 4810.6893146
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -4810.6891985, upper bound: 4810.6895125
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -4810.6903389, upper bound: 4810.6890127
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -4810.6899036, upper bound: 4810.6890127
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -4810.6898218, upper bound: 4810.6899538
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -4810.6905492, upper bound: 4810.6892914
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -4810.6892349, upper bound: 4810.6890127
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -4810.6890217, upper bound: 4810.6893146
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -4810.6891985, upper bound: 4810.6895125
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -4810.6895125, upper bound: 4810.6891985
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -4810.6893146, upper bound: 4810.6890217
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6892349
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -4810.6892914, upper bound: 4810.6905492
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -4810.6899538, upper bound: 4810.6898218
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6899036
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6903389
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -4810.6895125, upper bound: 4810.6891985
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -4810.6893146, upper bound: 4810.6890217
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6892349
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -4810.6892914, upper bound: 4810.6905492
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -4810.6899538, upper bound: 4810.6898218
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6899036
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6903389

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6897903, upper bound: 4810.6890127
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6900664, upper bound: 4810.6890127
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6897833, upper bound: 4810.6890127
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6898454, upper bound: 4810.6890127
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6897559, upper bound: 4810.6896986
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6897968, upper bound: 4810.6890127
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6896950, upper bound: 4810.6892914
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6900647, upper bound: 4810.6890127
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6892349, upper bound: 4810.6890127
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890217, upper bound: 4810.6893146
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6891985, upper bound: 4810.6895091
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6897903, upper bound: 4810.6890127
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6900664, upper bound: 4810.6890127
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6897833, upper bound: 4810.6890127
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6898454, upper bound: 4810.6890127
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6897559, upper bound: 4810.6896986
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6897968, upper bound: 4810.6890127
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6896950, upper bound: 4810.6892914
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6900647, upper bound: 4810.6890127
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6892349, upper bound: 4810.6890127
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890217, upper bound: 4810.6893146
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6891985, upper bound: 4810.6895091
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6895091, upper bound: 4810.6891985
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
time: 1.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6893146, upper bound: 4810.6890217
time: 0.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6892349
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6900647
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6892914, upper bound: 4810.6896950
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6897968
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6896986, upper bound: 4810.6897559
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6898454
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6897833
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6900664
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6897903
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
time: 1.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6895091, upper bound: 4810.6891985
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6893146, upper bound: 4810.6890217
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6892349
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6900647
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6892914, upper bound: 4810.6896950
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6897968
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6896986, upper bound: 4810.6897559
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6898454
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6897833
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6900664
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6897903
time: 0.64 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 4.25 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6897903, upper bound: 4810.6890127
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6900664, upper bound: 4810.6890127
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6897833, upper bound: 4810.6890127
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6898454, upper bound: 4810.6890127
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6897559, upper bound: 4810.6896986
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6897968, upper bound: 4810.6890127
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6896950, upper bound: 4810.6892914
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6900647, upper bound: 4810.6890127
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6892349, upper bound: 4810.6890127
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6890217, upper bound: 4810.6893146
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6891985, upper bound: 4810.6895091
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6897903, upper bound: 4810.6890127
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6900664, upper bound: 4810.6890127
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6897833, upper bound: 4810.6890127
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6898454, upper bound: 4810.6890127
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6897559, upper bound: 4810.6896986
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6897968, upper bound: 4810.6890127
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6896950, upper bound: 4810.6892914
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6900647, upper bound: 4810.6890127
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6892349, upper bound: 4810.6890127
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6890217, upper bound: 4810.6893146
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6891985, upper bound: 4810.6895091
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6895091, upper bound: 4810.6891985
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6893146, upper bound: 4810.6890217
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6892349
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6900647
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6892914, upper bound: 4810.6896950
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6897968
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6896986, upper bound: 4810.6897559
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6898454
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6897833
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6900664
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6897903
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6895091, upper bound: 4810.6891985
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6893146, upper bound: 4810.6890217
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6892349
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6900647
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6892914, upper bound: 4810.6896950
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6897968
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6896986, upper bound: 4810.6897559
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6898454
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6897833
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6900664
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6897903

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6882240, upper bound: 4810.6864652
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6881663, upper bound: 4810.6866366
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6889287, upper bound: 4810.6864652
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6881670, upper bound: 4810.6864652
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6881702, upper bound: 4810.6864652
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6880141, upper bound: 4810.6864652
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6882712, upper bound: 4810.6864652
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6873088, upper bound: 4810.6864652
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6864652
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6866882
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6864652
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6864652
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6881339, upper bound: 4810.6868407
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6879389, upper bound: 4810.6888662
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6881565, upper bound: 4810.6864652
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6871240, upper bound: 4810.6864652
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6878161, upper bound: 4810.6864652
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6880279, upper bound: 4810.6873708
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6892184, upper bound: 4810.6864652
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6881512, upper bound: 4810.6865845
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6864652
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6866276, upper bound: 4810.6865082
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6864652
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6864652
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6864652
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6874289
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6864652
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6864652
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6864652
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6865628, upper bound: 4810.6880823
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6864652
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6864652
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6882240, upper bound: 4810.6864652
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6881663, upper bound: 4810.6864652
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6889287, upper bound: 4810.6864652
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6881670, upper bound: 4810.6864652
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6881702, upper bound: 4810.6864652
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6880141, upper bound: 4810.6864652
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6882712, upper bound: 4810.6864652
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6873088, upper bound: 4810.6864652
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6864652
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6864652
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6864652
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6864652
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6881339, upper bound: 4810.6868407
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6879389, upper bound: 4810.6888662
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6881565, upper bound: 4810.6864652
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6871240, upper bound: 4810.6864652
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6878161, upper bound: 4810.6864652
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6880279, upper bound: 4810.6873708
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6892184, upper bound: 4810.6864652
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6881512, upper bound: 4810.6865845
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6868172, upper bound: 4810.6864652
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6868856, upper bound: 4810.6865082
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6869288, upper bound: 4810.6864652
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6868949, upper bound: 4810.6864652
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6864652
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6874289
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6864652
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6864652
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6867247, upper bound: 4810.6864652
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6867741, upper bound: 4810.6880823
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6868053, upper bound: 4810.6864652
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6867807, upper bound: 4810.6864652
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6867807
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6868053
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6880823, upper bound: 4810.6867741
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6867247
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6864652
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6864652
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 2.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6874289, upper bound: 4810.6864652
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6864652
time: 0.74 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 4.43 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6882240, upper bound: 4810.6864652
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6881663, upper bound: 4810.6866366
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6889287, upper bound: 4810.6864652
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6881670, upper bound: 4810.6864652
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6881702, upper bound: 4810.6864652
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6880141, upper bound: 4810.6864652
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6882712, upper bound: 4810.6864652
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6873088, upper bound: 4810.6864652
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6864652
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6866882
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6864652
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6864652
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6881339, upper bound: 4810.6868407
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6879389, upper bound: 4810.6888662
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6881565, upper bound: 4810.6864652
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6871240, upper bound: 4810.6864652
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6878161, upper bound: 4810.6864652
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6880279, upper bound: 4810.6873708
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6892184, upper bound: 4810.6864652
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6881512, upper bound: 4810.6865845
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6864652
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6866276, upper bound: 4810.6865082
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6864652
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6864652
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6864652
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6874289
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6864652
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6864652
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6864652
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6865628, upper bound: 4810.6880823
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6864652
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6864652
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6882240, upper bound: 4810.6864652
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6881663, upper bound: 4810.6864652
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6889287, upper bound: 4810.6864652
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6881670, upper bound: 4810.6864652
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6881702, upper bound: 4810.6864652
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6880141, upper bound: 4810.6864652
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6882712, upper bound: 4810.6864652
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6873088, upper bound: 4810.6864652
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6864652
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6864652
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6864652
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6864652
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6881339, upper bound: 4810.6868407
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6879389, upper bound: 4810.6888662
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6881565, upper bound: 4810.6864652
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6871240, upper bound: 4810.6864652
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6878161, upper bound: 4810.6864652
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6880279, upper bound: 4810.6873708
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6892184, upper bound: 4810.6864652
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6881512, upper bound: 4810.6865845
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6868172, upper bound: 4810.6864652
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6868856, upper bound: 4810.6865082
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6869288, upper bound: 4810.6864652
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6868949, upper bound: 4810.6864652
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6864652
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6874289
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6864652
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6864652
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6867247, upper bound: 4810.6864652
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6867741, upper bound: 4810.6880823
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6868053, upper bound: 4810.6864652
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6867807, upper bound: 4810.6864652
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6867807
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6868053
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6880823, upper bound: 4810.6867741
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6867247
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6864652
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6864652
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6874289, upper bound: 4810.6864652
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -4810.6864652, upper bound: 4810.6864652
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6892349
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6900647
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6892914, upper bound: 4810.6896950
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6897968
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6896986, upper bound: 4810.6897559
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6898454
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6897833
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6900664
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6897903
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6895091, upper bound: 4810.6891985
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6893146, upper bound: 4810.6890217
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6892349
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6900647
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6892914, upper bound: 4810.6896950
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6897968
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6896986, upper bound: 4810.6897559
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6890127
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6898454
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6897833
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6900664
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6890127, upper bound: 4810.6897903

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 4.65 + 416.34 = 420.99 seconds
