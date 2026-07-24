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
execution time: IAR + RelationalAnalysis = 1.02 + 1.77 = 2.79 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -4810.7054486, upper bound: 4810.7054486

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6864274, upper bound: 4810.6864274
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6864274, upper bound: 4810.6864274
time: 0.46 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.96 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.96
Output dim: 0, lower bound: -4810.6864274, upper bound: 4810.6864274
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.96
Output dim: 0, lower bound: -4810.6864274, upper bound: 4810.6864274

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6857569, upper bound: 4810.6864274
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6864274, upper bound: 4810.6857244
time: 0.49 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6864274, upper bound: 4810.6861775
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6862455, upper bound: 4810.6864274
time: 0.44 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.81 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.81
Output dim: 0, lower bound: -4810.6857569, upper bound: 4810.6864274
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.81
Output dim: 0, lower bound: -4810.6864274, upper bound: 4810.6857244
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.81
Output dim: 0, lower bound: -4810.6864274, upper bound: 4810.6861775
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.81
Output dim: 0, lower bound: -4810.6862455, upper bound: 4810.6864274

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6851484, upper bound: 4810.6864260
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6857567, upper bound: 4810.6863436
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6864274, upper bound: 4810.6855149
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6854168, upper bound: 4810.6857244
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6817432, upper bound: 4810.6807296
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6813248, upper bound: 4810.6813473
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6857835, upper bound: 4810.6852134
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6849195, upper bound: 4810.6859862
time: 0.49 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.02 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.02
Output dim: 0, lower bound: -4810.6851484, upper bound: 4810.6864260
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.02
Output dim: 0, lower bound: -4810.6857567, upper bound: 4810.6863436
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.02
Output dim: 0, lower bound: -4810.6864274, upper bound: 4810.6855149
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.02
Output dim: 0, lower bound: -4810.6854168, upper bound: 4810.6857244
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.02
Output dim: 0, lower bound: -4810.6817432, upper bound: 4810.6807296
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.02
Output dim: 0, lower bound: -4810.6813248, upper bound: 4810.6813473
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.02
Output dim: 0, lower bound: -4810.6857835, upper bound: 4810.6852134
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.02
Output dim: 0, lower bound: -4810.6849195, upper bound: 4810.6859862

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6829461, upper bound: 4810.6838512
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6829461, upper bound: 4810.6837474
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6814784, upper bound: 4810.6812592
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6811532, upper bound: 4810.6816058
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6856904, upper bound: 4810.6849475
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6852702, upper bound: 4810.6851899
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6845629, upper bound: 4810.6855833
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6852737, upper bound: 4810.6852975
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6796960, upper bound: 4810.6791586
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6796960, upper bound: 4810.6791586
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6812592, upper bound: 4810.6813446
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6813171, upper bound: 4810.6808088
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6857835, upper bound: 4810.6844214
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6849977, upper bound: 4810.6852067
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6843540, upper bound: 4810.6845715
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6844056, upper bound: 4810.6849900
time: 0.49 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 1.98 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.98
Output dim: 0, lower bound: -4810.6829461, upper bound: 4810.6838512
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.98
Output dim: 0, lower bound: -4810.6829461, upper bound: 4810.6837474
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.98
Output dim: 0, lower bound: -4810.6814784, upper bound: 4810.6812592
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.98
Output dim: 0, lower bound: -4810.6811532, upper bound: 4810.6816058
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.98
Output dim: 0, lower bound: -4810.6856904, upper bound: 4810.6849475
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.98
Output dim: 0, lower bound: -4810.6852702, upper bound: 4810.6851899
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.98
Output dim: 0, lower bound: -4810.6845629, upper bound: 4810.6855833
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.98
Output dim: 0, lower bound: -4810.6852737, upper bound: 4810.6852975
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.98
Output dim: 0, lower bound: -4810.6796960, upper bound: 4810.6791586
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.98
Output dim: 0, lower bound: -4810.6796960, upper bound: 4810.6791586
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.98
Output dim: 0, lower bound: -4810.6812592, upper bound: 4810.6813446
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.98
Output dim: 0, lower bound: -4810.6813171, upper bound: 4810.6808088
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.98
Output dim: 0, lower bound: -4810.6857835, upper bound: 4810.6844214
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.98
Output dim: 0, lower bound: -4810.6849977, upper bound: 4810.6852067
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.98
Output dim: 0, lower bound: -4810.6843540, upper bound: 4810.6845715
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.98
Output dim: 0, lower bound: -4810.6844056, upper bound: 4810.6849900

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6766334, upper bound: 4810.6770937
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6767191, upper bound: 4810.6770692
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6766334, upper bound: 4810.6770937
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6767191, upper bound: 4810.6770692
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6810209, upper bound: 4810.6808368
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6810722, upper bound: 4810.6808311
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6803063, upper bound: 4810.6800992
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6797869, upper bound: 4810.6805209
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6850003, upper bound: 4810.6833985
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6845050, upper bound: 4810.6842419
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6829935, upper bound: 4810.6820628
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6829935, upper bound: 4810.6820628
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6845067, upper bound: 4810.6855833
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6845629, upper bound: 4810.6855377
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6850546, upper bound: 4810.6834294
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6848764, upper bound: 4810.6847866
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6791384, upper bound: 4810.6773606
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6773606, upper bound: 4810.6785636
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6791076, upper bound: 4810.6774212
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6790641, upper bound: 4810.6785837
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6808311, upper bound: 4810.6809267
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6808368, upper bound: 4810.6807132
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6811532, upper bound: 4810.6808088
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6813171, upper bound: 4810.6804446
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6847856, upper bound: 4810.6842699
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6856279, upper bound: 4810.6838816
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6845158, upper bound: 4810.6850568
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6848457, upper bound: 4810.6842675
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6830560, upper bound: 4810.6844544
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6842599, upper bound: 4810.6828372
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6831519, upper bound: 4810.6841347
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6831432, upper bound: 4810.6841675
time: 0.60 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 2.12 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 0, lower bound: -4810.6766334, upper bound: 4810.6770937
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 0, lower bound: -4810.6767191, upper bound: 4810.6770692
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 0, lower bound: -4810.6766334, upper bound: 4810.6770937
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 0, lower bound: -4810.6767191, upper bound: 4810.6770692
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 0, lower bound: -4810.6810209, upper bound: 4810.6808368
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 0, lower bound: -4810.6810722, upper bound: 4810.6808311
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 0, lower bound: -4810.6803063, upper bound: 4810.6800992
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 0, lower bound: -4810.6797869, upper bound: 4810.6805209
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 0, lower bound: -4810.6850003, upper bound: 4810.6833985
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 0, lower bound: -4810.6845050, upper bound: 4810.6842419
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 0, lower bound: -4810.6829935, upper bound: 4810.6820628
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 0, lower bound: -4810.6829935, upper bound: 4810.6820628
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 0, lower bound: -4810.6845067, upper bound: 4810.6855833
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 0, lower bound: -4810.6845629, upper bound: 4810.6855377
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 0, lower bound: -4810.6850546, upper bound: 4810.6834294
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 0, lower bound: -4810.6848764, upper bound: 4810.6847866
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 0, lower bound: -4810.6791384, upper bound: 4810.6773606
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 0, lower bound: -4810.6773606, upper bound: 4810.6785636
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 0, lower bound: -4810.6791076, upper bound: 4810.6774212
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 0, lower bound: -4810.6790641, upper bound: 4810.6785837
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 0, lower bound: -4810.6808311, upper bound: 4810.6809267
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 0, lower bound: -4810.6808368, upper bound: 4810.6807132
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 0, lower bound: -4810.6811532, upper bound: 4810.6808088
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 0, lower bound: -4810.6813171, upper bound: 4810.6804446
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 0, lower bound: -4810.6847856, upper bound: 4810.6842699
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 0, lower bound: -4810.6856279, upper bound: 4810.6838816
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 0, lower bound: -4810.6845158, upper bound: 4810.6850568
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 0, lower bound: -4810.6848457, upper bound: 4810.6842675
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 0, lower bound: -4810.6830560, upper bound: 4810.6844544
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 0, lower bound: -4810.6842599, upper bound: 4810.6828372
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 0, lower bound: -4810.6831519, upper bound: 4810.6841347
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 0, lower bound: -4810.6831432, upper bound: 4810.6841675

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6741869, upper bound: 4810.6741869
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6741869, upper bound: 4810.6741869
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6759113, upper bound: 4810.6758649
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6758524, upper bound: 4810.6764601
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6766334, upper bound: 4810.6766362
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6766334, upper bound: 4810.6770829
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6767070, upper bound: 4810.6766249
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6766249, upper bound: 4810.6770692
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6810209, upper bound: 4810.6796585
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6806814, upper bound: 4810.6808368
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6790641, upper bound: 4810.6774545
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6790641, upper bound: 4810.6775478
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6741869, upper bound: 4810.6741999
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6741869, upper bound: 4810.6741999
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6791030, upper bound: 4810.6795693
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6791030, upper bound: 4810.6798962
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6841665, upper bound: 4810.6832875
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6848845, upper bound: 4810.6827689
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6819126, upper bound: 4810.6814521
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6814521, upper bound: 4810.6814521
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6819945, upper bound: 4810.6819945
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6828951, upper bound: 4810.6819945
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6826071, upper bound: 4810.6814521
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6821440, upper bound: 4810.6814521
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6840169, upper bound: 4810.6846978
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6834294, upper bound: 4810.6851588
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6843301, upper bound: 4810.6855377
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6845629, upper bound: 4810.6844319
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6829935, upper bound: 4810.6829935
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6837495, upper bound: 4810.6829935
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6841686, upper bound: 4810.6827689
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6834982, upper bound: 4810.6841786
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6763403, upper bound: 4810.6758524
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6763403, upper bound: 4810.6758524
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6773606, upper bound: 4810.6783833
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6773606, upper bound: 4810.6785636
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6760480, upper bound: 4810.6760480
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6760480, upper bound: 4810.6760480
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6790641, upper bound: 4810.6785837
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6787620, upper bound: 4810.6774212
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6799817, upper bound: 4810.6796778
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6793954, upper bound: 4810.6799773
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6802665, upper bound: 4810.6790161
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6794820, upper bound: 4810.6801089
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6767285, upper bound: 4810.6766334
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6767285, upper bound: 4810.6766334
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6808634, upper bound: 4810.6797162
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6795492, upper bound: 4810.6797033
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6825927, upper bound: 4810.6815046
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6825927, upper bound: 4810.6815046
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6831533, upper bound: 4810.6814467
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6831533, upper bound: 4810.6814467
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6814467, upper bound: 4810.6819529
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6814467, upper bound: 4810.6819617
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6848457, upper bound: 4810.6842675
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6847174, upper bound: 4810.6838816
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6830227, upper bound: 4810.6839259
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6827689, upper bound: 4810.6844544
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6790130, upper bound: 4810.6787154
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6788897, upper bound: 4810.6787154
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6831360, upper bound: 4810.6835273
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6825840, upper bound: 4810.6841347
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6830741, upper bound: 4810.6827092
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6828635, upper bound: 4810.6841675
time: 0.48 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.25 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6741869, upper bound: 4810.6741869
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6741869, upper bound: 4810.6741869
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6759113, upper bound: 4810.6758649
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6758524, upper bound: 4810.6764601
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6766334, upper bound: 4810.6766362
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6766334, upper bound: 4810.6770829
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6767070, upper bound: 4810.6766249
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6766249, upper bound: 4810.6770692
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6810209, upper bound: 4810.6796585
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6806814, upper bound: 4810.6808368
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6790641, upper bound: 4810.6774545
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6790641, upper bound: 4810.6775478
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6741869, upper bound: 4810.6741999
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6741869, upper bound: 4810.6741999
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6791030, upper bound: 4810.6795693
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6791030, upper bound: 4810.6798962
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6841665, upper bound: 4810.6832875
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6848845, upper bound: 4810.6827689
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6819126, upper bound: 4810.6814521
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6814521, upper bound: 4810.6814521
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6819945, upper bound: 4810.6819945
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6828951, upper bound: 4810.6819945
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6826071, upper bound: 4810.6814521
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6821440, upper bound: 4810.6814521
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6840169, upper bound: 4810.6846978
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6834294, upper bound: 4810.6851588
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6843301, upper bound: 4810.6855377
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6845629, upper bound: 4810.6844319
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6829935, upper bound: 4810.6829935
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6837495, upper bound: 4810.6829935
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6841686, upper bound: 4810.6827689
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6834982, upper bound: 4810.6841786
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6763403, upper bound: 4810.6758524
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6763403, upper bound: 4810.6758524
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6773606, upper bound: 4810.6783833
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6773606, upper bound: 4810.6785636
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6760480, upper bound: 4810.6760480
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6760480, upper bound: 4810.6760480
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6790641, upper bound: 4810.6785837
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6787620, upper bound: 4810.6774212
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6799817, upper bound: 4810.6796778
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6793954, upper bound: 4810.6799773
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6802665, upper bound: 4810.6790161
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6794820, upper bound: 4810.6801089
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6767285, upper bound: 4810.6766334
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6767285, upper bound: 4810.6766334
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6808634, upper bound: 4810.6797162
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6795492, upper bound: 4810.6797033
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6825927, upper bound: 4810.6815046
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6825927, upper bound: 4810.6815046
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6831533, upper bound: 4810.6814467
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6831533, upper bound: 4810.6814467
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6814467, upper bound: 4810.6819529
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6814467, upper bound: 4810.6819617
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6848457, upper bound: 4810.6842675
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6847174, upper bound: 4810.6838816
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6830227, upper bound: 4810.6839259
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6827689, upper bound: 4810.6844544
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6790130, upper bound: 4810.6787154
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6788897, upper bound: 4810.6787154
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6831360, upper bound: 4810.6835273
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6825840, upper bound: 4810.6841347
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6830741, upper bound: 4810.6827092
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -4810.6828635, upper bound: 4810.6841675

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6741869, upper bound: 4810.6741869
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6741869, upper bound: 4810.6741869
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6737599, upper bound: 4810.6737599
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6737599, upper bound: 4810.6737599
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6759113, upper bound: 4810.6758614
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6758524, upper bound: 4810.6758649
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6735435, upper bound: 4810.6735538
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6735435, upper bound: 4810.6735538
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6741869, upper bound: 4810.6741869
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6741869, upper bound: 4810.6741869
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6760480, upper bound: 4810.6762249
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6760480, upper bound: 4810.6764687
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6767070, upper bound: 4810.6766249
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6766249, upper bound: 4810.6766249
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6757393, upper bound: 4810.6757877
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6757393, upper bound: 4810.6764601
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6791076, upper bound: 4810.6774212
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6791076, upper bound: 4810.6774212
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6806729, upper bound: 4810.6804454
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6806814, upper bound: 4810.6808368
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6760480, upper bound: 4810.6760480
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6760480, upper bound: 4810.6760508
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6790641, upper bound: 4810.6774551
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6788938, upper bound: 4810.6775478
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6741869, upper bound: 4810.6741869
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6741869, upper bound: 4810.6741999
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6737599, upper bound: 4810.6737619
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6737599, upper bound: 4810.6737599
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6735435, upper bound: 4810.6735435
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6735435, upper bound: 4810.6735435
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6790363, upper bound: 4810.6790363
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6790363, upper bound: 4810.6798962
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6794853, upper bound: 4810.6785882
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6794555, upper bound: 4810.6787319
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6848766, upper bound: 4810.6827689
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6848845, upper bound: 4810.6827689
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6819126, upper bound: 4810.6814521
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6814521, upper bound: 4810.6814521
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6735435, upper bound: 4810.6735435
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6735435, upper bound: 4810.6735435
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6737599, upper bound: 4810.6737599
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6737599, upper bound: 4810.6737599
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6828682, upper bound: 4810.6819945
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6828951, upper bound: 4810.6819945
time: 1.04 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6735757, upper bound: 4810.6735435
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6735435, upper bound: 4810.6735435
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6821440, upper bound: 4810.6813456
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6813487, upper bound: 4810.6813456
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6829935, upper bound: 4810.6835333
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6833082, upper bound: 4810.6835333
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6834232, upper bound: 4810.6851545
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6834232, upper bound: 4810.6836430
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6820153, upper bound: 4810.6821956
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6820153, upper bound: 4810.6820153
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6841267, upper bound: 4810.6838803
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6838803, upper bound: 4810.6839233
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6819012, upper bound: 4810.6819012
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6819012, upper bound: 4810.6819012
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 25

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6829918, upper bound: 4810.6829918
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6837495, upper bound: 4810.6829918
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Candidate
type: DSZ, layer: 1, pos: 42

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6823783, upper bound: 4810.6823783
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6829613, upper bound: 4810.6823783
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6787587, upper bound: 4810.6793903
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6785882, upper bound: 4810.6794370
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Candidate
type: DSZ, layer: 1, pos: 25

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6758524, upper bound: 4810.6758524
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6763403, upper bound: 4810.6758524
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6735737, upper bound: 4810.6735435
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6735435, upper bound: 4810.6735435
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6773606, upper bound: 4810.6781914
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6773606, upper bound: 4810.6783833
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 42

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6773606, upper bound: 4810.6785636
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6773606, upper bound: 4810.6773606
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6753076, upper bound: 4810.6753076
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6753076, upper bound: 4810.6753076
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6753076, upper bound: 4810.6753076
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6753076, upper bound: 4810.6753076
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6790641, upper bound: 4810.6774853
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6783026, upper bound: 4810.6785837
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6760480, upper bound: 4810.6760480
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6760480, upper bound: 4810.6760480
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6775413, upper bound: 4810.6774213
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6774910, upper bound: 4810.6774213
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6737599, upper bound: 4810.6737602
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6737599, upper bound: 4810.6737599
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6757424, upper bound: 4810.6753076
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6754716, upper bound: 4810.6753076
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6789834, upper bound: 4810.6787925
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6787273, upper bound: 4810.6793553
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6762039, upper bound: 4810.6760480
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6760480, upper bound: 4810.6760480
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6758524, upper bound: 4810.6758677
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6758524, upper bound: 4810.6758722
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6787335, upper bound: 4810.6773606
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6773606, upper bound: 4810.6773606
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6773606, upper bound: 4810.6773606
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6773606, upper bound: 4810.6773606
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 42

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6824521, upper bound: 4810.6814829
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6825927, upper bound: 4810.6815046
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6821903, upper bound: 4810.6812959
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6812959, upper bound: 4810.6812959
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6755158, upper bound: 4810.6752494
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6755176, upper bound: 4810.6752494
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6831533, upper bound: 4810.6814467
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6831393, upper bound: 4810.6814467
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6814467, upper bound: 4810.6819529
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6814467, upper bound: 4810.6815692
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6814467, upper bound: 4810.6819617
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6814467, upper bound: 4810.6815692
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6824864, upper bound: 4810.6824864
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6835507, upper bound: 4810.6824933
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6824864, upper bound: 4810.6824864
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6833273, upper bound: 4810.6824864
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6813571, upper bound: 4810.6813571
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6813571, upper bound: 4810.6813571
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6827689, upper bound: 4810.6844544
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6827689, upper bound: 4810.6839723
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6767826, upper bound: 4810.6767826
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6767826, upper bound: 4810.6767826
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6767826, upper bound: 4810.6767826
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6767826, upper bound: 4810.6767826
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6826900, upper bound: 4810.6831641
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6831360, upper bound: 4810.6835273
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6823783, upper bound: 4810.6839970
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6823783, upper bound: 4810.6835313
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6829729, upper bound: 4810.6827068
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6830741, upper bound: 4810.6825829
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6773606, upper bound: 4810.6784891
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6773606, upper bound: 4810.6786086
time: 0.50 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 2.65 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6741869, upper bound: 4810.6741869
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6741869, upper bound: 4810.6741869
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6737599, upper bound: 4810.6737599
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6737599, upper bound: 4810.6737599
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6759113, upper bound: 4810.6758614
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6758524, upper bound: 4810.6758649
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6735435, upper bound: 4810.6735538
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6735435, upper bound: 4810.6735538
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6741869, upper bound: 4810.6741869
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6741869, upper bound: 4810.6741869
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6760480, upper bound: 4810.6762249
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6760480, upper bound: 4810.6764687
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6767070, upper bound: 4810.6766249
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6766249, upper bound: 4810.6766249
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6757393, upper bound: 4810.6757877
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6757393, upper bound: 4810.6764601
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6791076, upper bound: 4810.6774212
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6791076, upper bound: 4810.6774212
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6806729, upper bound: 4810.6804454
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6806814, upper bound: 4810.6808368
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6760480, upper bound: 4810.6760480
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6760480, upper bound: 4810.6760508
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6790641, upper bound: 4810.6774551
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6788938, upper bound: 4810.6775478
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6741869, upper bound: 4810.6741869
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6741869, upper bound: 4810.6741999
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6737599, upper bound: 4810.6737619
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6737599, upper bound: 4810.6737599
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6735435, upper bound: 4810.6735435
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6735435, upper bound: 4810.6735435
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6790363, upper bound: 4810.6790363
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6790363, upper bound: 4810.6798962
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6794853, upper bound: 4810.6785882
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6794555, upper bound: 4810.6787319
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6848766, upper bound: 4810.6827689
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6848845, upper bound: 4810.6827689
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6819126, upper bound: 4810.6814521
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6814521, upper bound: 4810.6814521
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6735435, upper bound: 4810.6735435
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6735435, upper bound: 4810.6735435
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6737599, upper bound: 4810.6737599
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6737599, upper bound: 4810.6737599
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6828682, upper bound: 4810.6819945
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6828951, upper bound: 4810.6819945
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6735757, upper bound: 4810.6735435
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6735435, upper bound: 4810.6735435
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6821440, upper bound: 4810.6813456
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6813487, upper bound: 4810.6813456
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6829935, upper bound: 4810.6835333
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6833082, upper bound: 4810.6835333
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6834232, upper bound: 4810.6851545
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6834232, upper bound: 4810.6836430
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6820153, upper bound: 4810.6821956
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6820153, upper bound: 4810.6820153
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6841267, upper bound: 4810.6838803
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6838803, upper bound: 4810.6839233
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6819012, upper bound: 4810.6819012
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6819012, upper bound: 4810.6819012
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6829918, upper bound: 4810.6829918
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6837495, upper bound: 4810.6829918
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6823783, upper bound: 4810.6823783
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6829613, upper bound: 4810.6823783
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6787587, upper bound: 4810.6793903
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6785882, upper bound: 4810.6794370
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6758524, upper bound: 4810.6758524
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6763403, upper bound: 4810.6758524
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6735737, upper bound: 4810.6735435
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6735435, upper bound: 4810.6735435
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6773606, upper bound: 4810.6781914
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6773606, upper bound: 4810.6783833
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6773606, upper bound: 4810.6785636
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6773606, upper bound: 4810.6773606
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6753076, upper bound: 4810.6753076
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6753076, upper bound: 4810.6753076
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6753076, upper bound: 4810.6753076
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6753076, upper bound: 4810.6753076
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6790641, upper bound: 4810.6774853
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6783026, upper bound: 4810.6785837
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6760480, upper bound: 4810.6760480
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6760480, upper bound: 4810.6760480
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6775413, upper bound: 4810.6774213
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6774910, upper bound: 4810.6774213
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6737599, upper bound: 4810.6737602
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6737599, upper bound: 4810.6737599
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6757424, upper bound: 4810.6753076
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6754716, upper bound: 4810.6753076
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6789834, upper bound: 4810.6787925
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6787273, upper bound: 4810.6793553
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6762039, upper bound: 4810.6760480
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6760480, upper bound: 4810.6760480
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6758524, upper bound: 4810.6758677
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6758524, upper bound: 4810.6758722
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6787335, upper bound: 4810.6773606
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6773606, upper bound: 4810.6773606
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6773606, upper bound: 4810.6773606
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6773606, upper bound: 4810.6773606
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6824521, upper bound: 4810.6814829
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6825927, upper bound: 4810.6815046
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6821903, upper bound: 4810.6812959
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6812959, upper bound: 4810.6812959
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6755158, upper bound: 4810.6752494
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6755176, upper bound: 4810.6752494
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6831533, upper bound: 4810.6814467
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6831393, upper bound: 4810.6814467
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6814467, upper bound: 4810.6819529
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6814467, upper bound: 4810.6815692
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6814467, upper bound: 4810.6819617
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6814467, upper bound: 4810.6815692
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6824864, upper bound: 4810.6824864
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6835507, upper bound: 4810.6824933
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6824864, upper bound: 4810.6824864
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6833273, upper bound: 4810.6824864
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6813571, upper bound: 4810.6813571
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6813571, upper bound: 4810.6813571
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6827689, upper bound: 4810.6844544
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6827689, upper bound: 4810.6839723
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6767826, upper bound: 4810.6767826
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6767826, upper bound: 4810.6767826
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6767826, upper bound: 4810.6767826
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6767826, upper bound: 4810.6767826
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6826900, upper bound: 4810.6831641
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6831360, upper bound: 4810.6835273
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6823783, upper bound: 4810.6839970
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6823783, upper bound: 4810.6835313
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6829729, upper bound: 4810.6827068
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6830741, upper bound: 4810.6825829
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6773606, upper bound: 4810.6784891
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.65
Output dim: 0, lower bound: -4810.6773606, upper bound: 4810.6786086

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6741869, upper bound: 4810.6741869
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6741869, upper bound: 4810.6741869
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6737599, upper bound: 4810.6737599
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6737599, upper bound: 4810.6737599
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6737599, upper bound: 4810.6737599
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6737599, upper bound: 4810.6737599
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6737599, upper bound: 4810.6737599
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6737599, upper bound: 4810.6737599
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6753461, upper bound: 4810.6753076
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6753076, upper bound: 4810.6753076
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6753155, upper bound: 4810.6753076
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6753076, upper bound: 4810.6753076
time: 3.00 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6730194, upper bound: 4810.6730394
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6730194, upper bound: 4810.6730194
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6735435, upper bound: 4810.6735435
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6735435, upper bound: 4810.6735538
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6741869, upper bound: 4810.6741869
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6741869, upper bound: 4810.6741869
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6735435, upper bound: 4810.6735435
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6735435, upper bound: 4810.6735435
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6760480, upper bound: 4810.6760480
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6760480, upper bound: 4810.6762249
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6753076, upper bound: 4810.6753076
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6753076, upper bound: 4810.6757757
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6758963, upper bound: 4810.6757393
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6757393, upper bound: 4810.6757393
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6760480, upper bound: 4810.6760480
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6760480, upper bound: 4810.6760480
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6757393, upper bound: 4810.6757877
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6757393, upper bound: 4810.6757393
time: 0.98 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6752494, upper bound: 4810.6757768
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6752494, upper bound: 4810.6752843
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6786671, upper bound: 4810.6774212
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6774212, upper bound: 4810.6774212
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6791076, upper bound: 4810.6774212
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6787535, upper bound: 4810.6774212
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6774212, upper bound: 4810.6774212
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6785815, upper bound: 4810.6774212
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6800682, upper bound: 4810.6794820
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6788546, upper bound: 4810.6802665
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6753076, upper bound: 4810.6753076
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6753076, upper bound: 4810.6753076
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6737599, upper bound: 4810.6737599
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6737599, upper bound: 4810.6737599
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6786679, upper bound: 4810.6774212
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6774212, upper bound: 4810.6774494
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6788938, upper bound: 4810.6774640
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6784946, upper bound: 4810.6775478
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6737599, upper bound: 4810.6737599
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6737599, upper bound: 4810.6737599
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 42

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6741869, upper bound: 4810.6741869
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6741869, upper bound: 4810.6741999
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6730194, upper bound: 4810.6730194
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6730194, upper bound: 4810.6730689
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6730194, upper bound: 4810.6730194
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6730194, upper bound: 4810.6730194
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6730194, upper bound: 4810.6730194
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6730194, upper bound: 4810.6730194
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6735435, upper bound: 4810.6735435
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6735435, upper bound: 4810.6735435
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6785824, upper bound: 4810.6785824
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6785824, upper bound: 4810.6785824
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6785824, upper bound: 4810.6795149
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6785824, upper bound: 4810.6791521
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6794001, upper bound: 4810.6785824
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6794781, upper bound: 4810.6785824
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6730194, upper bound: 4810.6730194
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6730194, upper bound: 4810.6730194
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6848024, upper bound: 4810.6827625
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6848725, upper bound: 4810.6827625
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6792405, upper bound: 4810.6785882
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6785882, upper bound: 4810.6785882
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6819126, upper bound: 4810.6813456
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6813456, upper bound: 4810.6813456
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6813571, upper bound: 4810.6813571
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6813571, upper bound: 4810.6813571
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6735435, upper bound: 4810.6735435
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6735435, upper bound: 4810.6735435
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6735435, upper bound: 4810.6735435
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6735435, upper bound: 4810.6735435
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6737599, upper bound: 4810.6737599
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6737599, upper bound: 4810.6737599
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6737599, upper bound: 4810.6737599
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6737599, upper bound: 4810.6737599
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6824529, upper bound: 4810.6813571
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6819617, upper bound: 4810.6813571
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6828263, upper bound: 4810.6819012
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6819032, upper bound: 4810.6819012
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6735732, upper bound: 4810.6735435
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6735757, upper bound: 4810.6735435
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6735435, upper bound: 4810.6735435
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6735435, upper bound: 4810.6735435
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6819684, upper bound: 4810.6813456
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6821440, upper bound: 4810.6813456
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6735435, upper bound: 4810.6735435
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6735435, upper bound: 4810.6735435
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6774212, upper bound: 4810.6774212
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6776298, upper bound: 4810.6774212
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6828020, upper bound: 4810.6825505
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6823783, upper bound: 4810.6829279
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6827625, upper bound: 4810.6827625
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6827625, upper bound: 4810.6844700
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6819945, upper bound: 4810.6819945
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6819945, upper bound: 4810.6819945
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6814467, upper bound: 4810.6814467
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6814467, upper bound: 4810.6815790
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6760480, upper bound: 4810.6760480
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6760480, upper bound: 4810.6760480
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6814467, upper bound: 4810.6814467
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6814467, upper bound: 4810.6814467
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6788546, upper bound: 4810.6789763
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6788546, upper bound: 4810.6790461
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6819012, upper bound: 4810.6819012
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6819012, upper bound: 4810.6819012
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062
1: -294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011
2: -466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004
3: -542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918
4: -407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Candidate
type: DSZ, layer: 1, pos: 42

### Candidate
type: DSZ, layer: 1, pos: 2

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6819012, upper bound: 4810.6819012
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6819012, upper bound: 4810.6819012
time: 0.48 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 2.64 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6741869, upper bound: 4810.6741869
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6741869, upper bound: 4810.6741869
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6737599, upper bound: 4810.6737599
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6737599, upper bound: 4810.6737599
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6737599, upper bound: 4810.6737599
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6737599, upper bound: 4810.6737599
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6737599, upper bound: 4810.6737599
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6737599, upper bound: 4810.6737599
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6753461, upper bound: 4810.6753076
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6753076, upper bound: 4810.6753076
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6753155, upper bound: 4810.6753076
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6753076, upper bound: 4810.6753076
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6730194, upper bound: 4810.6730394
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6730194, upper bound: 4810.6730194
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6735435, upper bound: 4810.6735435
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6735435, upper bound: 4810.6735538
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6741869, upper bound: 4810.6741869
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6741869, upper bound: 4810.6741869
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6735435, upper bound: 4810.6735435
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6735435, upper bound: 4810.6735435
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6760480, upper bound: 4810.6760480
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6760480, upper bound: 4810.6762249
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6753076, upper bound: 4810.6753076
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6753076, upper bound: 4810.6757757
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6758963, upper bound: 4810.6757393
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6757393, upper bound: 4810.6757393
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6760480, upper bound: 4810.6760480
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6760480, upper bound: 4810.6760480
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6757393, upper bound: 4810.6757877
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6757393, upper bound: 4810.6757393
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6752494, upper bound: 4810.6757768
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6752494, upper bound: 4810.6752843
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6786671, upper bound: 4810.6774212
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6774212, upper bound: 4810.6774212
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6791076, upper bound: 4810.6774212
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6787535, upper bound: 4810.6774212
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6774212, upper bound: 4810.6774212
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6785815, upper bound: 4810.6774212
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6800682, upper bound: 4810.6794820
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6788546, upper bound: 4810.6802665
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6753076, upper bound: 4810.6753076
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6753076, upper bound: 4810.6753076
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6737599, upper bound: 4810.6737599
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6737599, upper bound: 4810.6737599
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6786679, upper bound: 4810.6774212
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6774212, upper bound: 4810.6774494
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6788938, upper bound: 4810.6774640
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6784946, upper bound: 4810.6775478
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6737599, upper bound: 4810.6737599
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6737599, upper bound: 4810.6737599
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6741869, upper bound: 4810.6741869
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6741869, upper bound: 4810.6741999
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6730194, upper bound: 4810.6730194
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6730194, upper bound: 4810.6730689
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6730194, upper bound: 4810.6730194
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6730194, upper bound: 4810.6730194
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6730194, upper bound: 4810.6730194
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6730194, upper bound: 4810.6730194
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6735435, upper bound: 4810.6735435
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6735435, upper bound: 4810.6735435
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6785824, upper bound: 4810.6785824
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6785824, upper bound: 4810.6785824
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6785824, upper bound: 4810.6795149
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6785824, upper bound: 4810.6791521
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6794001, upper bound: 4810.6785824
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6794781, upper bound: 4810.6785824
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6730194, upper bound: 4810.6730194
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6730194, upper bound: 4810.6730194
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6848024, upper bound: 4810.6827625
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6848725, upper bound: 4810.6827625
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6792405, upper bound: 4810.6785882
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6785882, upper bound: 4810.6785882
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6819126, upper bound: 4810.6813456
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6813456, upper bound: 4810.6813456
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6813571, upper bound: 4810.6813571
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6813571, upper bound: 4810.6813571
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6735435, upper bound: 4810.6735435
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6735435, upper bound: 4810.6735435
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6735435, upper bound: 4810.6735435
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6735435, upper bound: 4810.6735435
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6737599, upper bound: 4810.6737599
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6737599, upper bound: 4810.6737599
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6737599, upper bound: 4810.6737599
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6737599, upper bound: 4810.6737599
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6824529, upper bound: 4810.6813571
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6819617, upper bound: 4810.6813571
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6828263, upper bound: 4810.6819012
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6819032, upper bound: 4810.6819012
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6735732, upper bound: 4810.6735435
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6735757, upper bound: 4810.6735435
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6735435, upper bound: 4810.6735435
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6735435, upper bound: 4810.6735435
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6819684, upper bound: 4810.6813456
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6821440, upper bound: 4810.6813456
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6735435, upper bound: 4810.6735435
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6735435, upper bound: 4810.6735435
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6774212, upper bound: 4810.6774212
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6776298, upper bound: 4810.6774212
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6828020, upper bound: 4810.6825505
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6823783, upper bound: 4810.6829279
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6827625, upper bound: 4810.6827625
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6827625, upper bound: 4810.6844700
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6819945, upper bound: 4810.6819945
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6819945, upper bound: 4810.6819945
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6814467, upper bound: 4810.6814467
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6814467, upper bound: 4810.6815790
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6760480, upper bound: 4810.6760480
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6760480, upper bound: 4810.6760480
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6814467, upper bound: 4810.6814467
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6814467, upper bound: 4810.6814467
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6788546, upper bound: 4810.6789763
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6788546, upper bound: 4810.6790461
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6819012, upper bound: 4810.6819012
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6819012, upper bound: 4810.6819012
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6819012, upper bound: 4810.6819012
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 0, lower bound: -4810.6819012, upper bound: 4810.6819012
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6829918, upper bound: 4810.6829918
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6837495, upper bound: 4810.6829918
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6823783, upper bound: 4810.6823783
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6829613, upper bound: 4810.6823783
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6787587, upper bound: 4810.6793903
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6785882, upper bound: 4810.6794370
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6758524, upper bound: 4810.6758524
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6763403, upper bound: 4810.6758524
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6735737, upper bound: 4810.6735435
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6735435, upper bound: 4810.6735435
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6773606, upper bound: 4810.6781914
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6773606, upper bound: 4810.6783833
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6773606, upper bound: 4810.6785636
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6773606, upper bound: 4810.6773606
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6753076, upper bound: 4810.6753076
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6753076, upper bound: 4810.6753076
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6753076, upper bound: 4810.6753076
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6753076, upper bound: 4810.6753076
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6790641, upper bound: 4810.6774853
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6783026, upper bound: 4810.6785837
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6760480, upper bound: 4810.6760480
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6760480, upper bound: 4810.6760480
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6775413, upper bound: 4810.6774213
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6774910, upper bound: 4810.6774213
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6737599, upper bound: 4810.6737602
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6737599, upper bound: 4810.6737599
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6757424, upper bound: 4810.6753076
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6754716, upper bound: 4810.6753076
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6789834, upper bound: 4810.6787925
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6787273, upper bound: 4810.6793553
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6762039, upper bound: 4810.6760480
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6760480, upper bound: 4810.6760480
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6758524, upper bound: 4810.6758677
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6758524, upper bound: 4810.6758722
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6787335, upper bound: 4810.6773606
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6773606, upper bound: 4810.6773606
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6773606, upper bound: 4810.6773606
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6773606, upper bound: 4810.6773606
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6824521, upper bound: 4810.6814829
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6825927, upper bound: 4810.6815046
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6821903, upper bound: 4810.6812959
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6812959, upper bound: 4810.6812959
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6755158, upper bound: 4810.6752494
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6755176, upper bound: 4810.6752494
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6831533, upper bound: 4810.6814467
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6831393, upper bound: 4810.6814467
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6814467, upper bound: 4810.6819529
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6814467, upper bound: 4810.6815692
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6814467, upper bound: 4810.6819617
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6814467, upper bound: 4810.6815692
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6824864, upper bound: 4810.6824864
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6835507, upper bound: 4810.6824933
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6824864, upper bound: 4810.6824864
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6833273, upper bound: 4810.6824864
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6813571, upper bound: 4810.6813571
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6813571, upper bound: 4810.6813571
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6827689, upper bound: 4810.6844544
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6827689, upper bound: 4810.6839723
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6767826, upper bound: 4810.6767826
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6767826, upper bound: 4810.6767826
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6767826, upper bound: 4810.6767826
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6767826, upper bound: 4810.6767826
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6826900, upper bound: 4810.6831641
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6831360, upper bound: 4810.6835273
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6823783, upper bound: 4810.6839970
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6823783, upper bound: 4810.6835313
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6829729, upper bound: 4810.6827068
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6830741, upper bound: 4810.6825829
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6773606, upper bound: 4810.6784891
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -4810.6773606, upper bound: 4810.6786086

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 2.79 + 417.98 = 420.77 seconds
