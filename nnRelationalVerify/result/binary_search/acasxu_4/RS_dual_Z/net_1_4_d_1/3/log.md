## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_4.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 0.07398656999999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144)
1: (-0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303)
2: (-0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905)
3: (-0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251)
4: (-0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503)

## BASE Result
execution time: IAR + LP analysis = 1.84 + 0.88 = 2.72 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0754965, upper bound: 0.0754965


# Binary Search by BASE starts (time budget: 1197.28 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.0909091


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.0909091, mid=0.0909091, abs_max=0.09181444346904755
rel_dist={0: [-0.07549650593465267, 0.07549650593465268]}

## Binary search (step 1) starts
Candidate diff: 0.0454545


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0454545, mid=0.0454545, abs_max=0.09181444346904755
rel_dist={0: [-0.074817289964232, 0.07481728996423198]}

## Binary search (step 2) starts
Candidate diff: 0.0227273


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0227273, mid=0.0227273, abs_max=0.09181444346904755
rel_dist={0: [-0.07418591474870181, 0.07418591474870184]}

## Binary search (step 3) starts
Candidate diff: 0.0113636


## IAR start
Binary search (step 3): status=Status.VERIFIED, low=0.0113636, high=0.0227273, mid=0.0113636, abs_max=0.09181444346904755
rel_dist={0: [-0.07385188413801798, 0.07385188413801799]}

## Binary search (step 4) starts
Candidate diff: 0.0170455


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0113636, high=0.0170455, mid=0.0170455, abs_max=0.09181444346904755
rel_dist={0: [-0.07401889944335989, 0.07401889944335993]}

## Binary search (step 5) starts
Candidate diff: 0.0142045


## IAR start
Binary search (step 5): status=Status.VERIFIED, low=0.0142045, high=0.0170455, mid=0.0142045, abs_max=0.09181444346904755
rel_dist={0: [-0.07393539179068895, 0.07393539179068898]}

## Binary search (step 6) starts
Candidate diff: 0.0156250


## IAR start
Binary search (step 6): status=Status.VERIFIED, low=0.0156250, high=0.0170455, mid=0.0156250, abs_max=0.09181444346904755
rel_dist={0: [-0.0739771456033364, 0.07397714560333646]}

## Binary search (step 7) starts
Candidate diff: 0.0163352


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0156250, high=0.0163352, mid=0.0163352, abs_max=0.09181444346904755
rel_dist={0: [-0.07399802255072409, 0.07399802255072413]}

## Binary search (step 8) starts
Candidate diff: 0.0159801


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0156250, high=0.0159801, mid=0.0159801, abs_max=0.09181444346904755
rel_dist={0: [-0.0739875840496543, 0.07398758404965433]}

## Binary search (step 9) starts
Candidate diff: 0.0158026


## IAR start
Binary search (step 9): status=Status.VERIFIED, low=0.0158026, high=0.0159801, mid=0.0158026, abs_max=0.09181444346904755
rel_dist={0: [-0.07398236482649538, 0.0739823648264954]}

## Binary search (step 10) starts
Candidate diff: 0.0158913


## IAR start
Binary search (step 10): status=Status.VERIFIED, low=0.0158913, high=0.0159801, mid=0.0158913, abs_max=0.09181444346904755
rel_dist={0: [-0.07398497446545078, 0.07398497446545083]}

## Binary search (step 11) starts
Candidate diff: 0.0159357


## IAR start
Binary search (step 11): status=Status.VERIFIED, low=0.0159357, high=0.0159801, mid=0.0159357, abs_max=0.09181444346904755
rel_dist={0: [-0.07398627925755258, 0.07398627925755258]}

## Binary search (step 12) starts
Candidate diff: 0.0159579


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0159357, high=0.0159579, mid=0.0159579, abs_max=0.09181444346904755
rel_dist={0: [-0.07398693168097939, 0.07398693168097939]}

## Binary search (step 13) starts
Candidate diff: 0.0159468


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0159357, high=0.0159468, mid=0.0159468, abs_max=0.09181444346904755
rel_dist={0: [-0.07398660546926598, 0.07398660546926597]}

## Binary search (step 14) starts
Candidate diff: 0.0159413


## IAR start
Binary search (step 14): status=Status.VERIFIED, low=0.0159413, high=0.0159468, mid=0.0159413, abs_max=0.09181444346904755
rel_dist={0: [-0.07398644236340933, 0.07398644236340929]}

## Binary search (step 15) starts
Candidate diff: 0.0159440


## IAR start
Binary search (step 15): status=Status.VERIFIED, low=0.0159440, high=0.0159468, mid=0.0159440, abs_max=0.09181444346904755
rel_dist={0: [-0.0739865239437136, 0.07398652394371358]}

## Binary search (step 16) starts
Candidate diff: 0.0159454


## IAR start
Binary search (step 16): status=Status.VERIFIED, low=0.0159454, high=0.0159468, mid=0.0159454, abs_max=0.09181444346904755
rel_dist={0: [-0.07398656467911387, 0.07398656467911383]}

## Binary search (step 17) starts
Candidate diff: 0.0159461


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0159454, high=0.0159461, mid=0.0159461, abs_max=0.09181444346904755
rel_dist={0: [-0.07398658510156589, 0.07398658510156586]}

## Binary Search Result
Binary search time: 50.29 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01594543504552348


# Relational Split (RS_dual_Z) starts
Time budget: 1146.99 seconds

## Binary search (step 0) starts
Candidate diff: 0.0988818


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753536
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753784
time: 0.30 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.77 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.77
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753536
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.77
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753784

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0739197
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
time: 0.30 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0734268
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233
time: 0.31 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.57 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.57
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0739197
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.57
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.57
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0734268
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.57
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
time: 0.29 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.54 seconds
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.54
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.54
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.54
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.54
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749547
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749518
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749836
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749810
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.32 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0740920
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733790, upper bound: 0.0749010
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749766
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0735295, upper bound: 0.0749765
time: 0.33 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.87 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749547
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749518
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749836
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749810
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0740920
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 0, lower bound: -0.0733790, upper bound: 0.0749010
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749766
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 0, lower bound: -0.0735295, upper bound: 0.0749765

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721131
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722143
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721260
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722157
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724708
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724852
time: 0.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721101
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721101
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721224, upper bound: 0.0721463
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721224, upper bound: 0.0722027
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
time: 0.29 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.58 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721131
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722143
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721260
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722157
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724708
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724852
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721101
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721101
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 0, lower bound: -0.0721224, upper bound: 0.0721463
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 0, lower bound: -0.0721224, upper bound: 0.0722027
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
Binary search (step 0): status=Status.VERIFIED, low=0.0988818, high=0.1818182, mid=0.0988818, abs_max=0.09181444346904755
rel_dist={0: [-0.07549650593465267, 0.07549650593465268]}

## Binary search (step 1) starts
Candidate diff: 0.1403500


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753536
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753784
time: 0.29 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.76 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.76
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753536
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.76
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753784

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0739339
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
time: 0.31 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753228, upper bound: 0.0734268
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233
time: 0.31 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.51 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.51
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0739339
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.51
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.51
Output dim: 0, lower bound: -0.0753228, upper bound: 0.0734268
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.51
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0733262
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0733124
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961
time: 0.29 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.56 seconds
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.56
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.56
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.56
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0733262
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.56
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0733124
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.56
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.56
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749547
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749518
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749836
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749810
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0740920
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749010
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749766
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749765
time: 0.31 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.76 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749547
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749518
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749836
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749810
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0740920
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749010
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749766
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749765

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721131
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722143
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721260
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722157
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724708
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724852
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0733572
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0736225
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0740879
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730122, upper bound: 0.0744961
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
time: 0.30 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.61 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.61
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721131
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.61
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722143
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.61
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721260
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.61
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722157
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.61
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.61
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.61
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724708
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.61
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724852
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.61
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0733572
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.61
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0736225
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0740879
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -0.0730122, upper bound: 0.0744961
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.61
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.61
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.61
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.61
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0729548
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728096, upper bound: 0.0731555
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0727585
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727940, upper bound: 0.0732237
time: 0.35 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.06 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.06
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0729548
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.06
Output dim: 0, lower bound: -0.0728096, upper bound: 0.0731555
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.06
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0727585
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.06
Output dim: 0, lower bound: -0.0727940, upper bound: 0.0732237
Binary search (step 1): status=Status.VERIFIED, low=0.1403500, high=0.1818182, mid=0.1403500, abs_max=0.09181444346904755
rel_dist={0: [-0.07549650593465267, 0.07549650593465268]}

## Binary search (step 2) starts
Candidate diff: 0.1610841


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753536
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753784
time: 0.30 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.77 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.77
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753536
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.77
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753784

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0739373
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
time: 0.31 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753228, upper bound: 0.0734268
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233
time: 0.29 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.55 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.55
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0739373
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.55
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.55
Output dim: 0, lower bound: -0.0753228, upper bound: 0.0734268
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.55
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0733262
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0733124
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961
time: 0.29 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.54 seconds
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.54
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.54
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.54
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0733262
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.54
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0733124
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.54
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.54
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749547
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749518
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.32 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749836
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749810
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.32 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0740920
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749010
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.32 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749766
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749765
time: 0.32 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.92 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.92
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749547
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.92
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749518
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.92
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749836
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.92
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749810
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.92
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0740920
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.92
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749010
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.92
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749766
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.92
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749765

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721131
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722143
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721260
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722157
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724708
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724852
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0733572
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0736225
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0728390
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730122, upper bound: 0.0744961
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
time: 0.32 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.86 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721131
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722143
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721260
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722157
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724708
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724852
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0733572
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0736225
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0728390
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0730122, upper bound: 0.0744961
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0727585
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0732237
time: 0.33 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.32 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0727585
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0732237
Binary search (step 2): status=Status.VERIFIED, low=0.1610841, high=0.1818182, mid=0.1610841, abs_max=0.09181444346904755
rel_dist={0: [-0.07549650593465267, 0.07549650593465268]}

## Binary search (step 3) starts
Candidate diff: 0.1714511


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753536
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753784
time: 0.32 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.82 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.82
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753536
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.82
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753784

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0739373
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
time: 0.31 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0734268
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233
time: 0.31 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.81 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.81
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0739373
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.81
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.81
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0734268
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.81
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0736330, upper bound: 0.0750961
time: 0.32 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.80 seconds
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.80
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.80
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.80
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.80
Output dim: 0, lower bound: -0.0736330, upper bound: 0.0750961

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.33 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749547
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731370, upper bound: 0.0749518
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.33 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749836
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749810
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.33 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0740920
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749010
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.33 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749766
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749765
time: 0.33 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.99 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749547
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 0, lower bound: -0.0731370, upper bound: 0.0749518
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749836
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749810
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0740920
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749010
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749766
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749765

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721131
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722143
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721260
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722157
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724708
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724852
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721101
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721101
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721224, upper bound: 0.0721463
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721224, upper bound: 0.0722027
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
time: 0.32 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.86 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721131
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722143
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721260
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722157
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724708
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724852
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721101
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721101
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0721224, upper bound: 0.0721463
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0721224, upper bound: 0.0722027
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.86
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
Binary search (step 3): status=Status.VERIFIED, low=0.1714511, high=0.1818182, mid=0.1714511, abs_max=0.09181444346904755
rel_dist={0: [-0.07549650593465267, 0.07549650593465268]}

## Binary search (step 4) starts
Candidate diff: 0.1766347


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753536
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753784
time: 0.33 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.83 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.83
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753536
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.83
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753784

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0752233, upper bound: 0.0739373
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
time: 0.30 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0734268
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233
time: 0.31 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.81 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.81
Output dim: 0, lower bound: -0.0752233, upper bound: 0.0739373
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.81
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.81
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0734268
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.81
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0736330
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0734262
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961
time: 0.30 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.79 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.79
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0736330
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.79
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0734262
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.33 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749547
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749518
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749836
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749810
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0740920
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749010
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749766
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749765
time: 0.30 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.77 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749547
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749518
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749836
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749810
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0740920
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749010
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749766
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749765

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721131
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722143
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721260
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722157
time: 0.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724708
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724852
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721101
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721101
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721224, upper bound: 0.0721463
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721224, upper bound: 0.0722027
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
time: 0.31 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.63 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721131
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722143
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721260
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722157
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724708
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724852
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721101
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721101
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721224, upper bound: 0.0721463
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721224, upper bound: 0.0722027
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
Binary search (step 4): status=Status.VERIFIED, low=0.1766347, high=0.1818182, mid=0.1766347, abs_max=0.09181444346904755
rel_dist={0: [-0.07549650593465267, 0.07549650593465268]}

## Binary search (step 5) starts
Candidate diff: 0.1792264


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753536
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753784
time: 0.30 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.78 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.78
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753536
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.78
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753784

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0739373
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
time: 0.29 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753228, upper bound: 0.0734268
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233
time: 0.30 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.59 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.59
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0739373
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.59
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.59
Output dim: 0, lower bound: -0.0753228, upper bound: 0.0734268
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.59
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0733262
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0733124
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961
time: 0.29 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.55 seconds
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.55
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.55
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.55
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0733262
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.55
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0733124
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.55
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.55
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749547
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749518
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749836
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749810
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0740920
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749010
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749766
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0735295, upper bound: 0.0749765
time: 0.31 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.69 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749547
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749518
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749836
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749810
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0740920
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749010
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749766
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 0, lower bound: -0.0735295, upper bound: 0.0749765

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721131
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722143
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721260
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722157
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724708
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724852
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721101
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721101
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721224, upper bound: 0.0721463
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721224, upper bound: 0.0722027
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
time: 0.31 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.63 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721131
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722143
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721260
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722157
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724708
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724852
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721101
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721101
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721224, upper bound: 0.0721463
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721224, upper bound: 0.0722027
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
Binary search (step 5): status=Status.VERIFIED, low=0.1792264, high=0.1818182, mid=0.1792264, abs_max=0.09181444346904755
rel_dist={0: [-0.07549650593465267, 0.07549650593465268]}

## Binary search (step 6) starts
Candidate diff: 0.1805223


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753536
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753784
time: 0.31 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.78 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.78
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753536
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.78
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753784

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0739373
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
time: 0.29 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0734268
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233
time: 0.29 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.52 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.52
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0739373
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.52
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.52
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0734268
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.52
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961
time: 0.28 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.50 seconds
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.50
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.50
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.50
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.50
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749547
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749518
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749836
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749810
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0740920
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749010
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749766
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0735295, upper bound: 0.0749765
time: 0.31 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.75 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749547
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749518
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749836
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749810
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0740920
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749010
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749766
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 0, lower bound: -0.0735295, upper bound: 0.0749765

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721131
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722143
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721260
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722157
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724708
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724852
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0733572
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0736225
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0740879
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0744961
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
time: 0.31 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.62 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721131
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722143
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721260
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722157
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724708
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724852
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0733572
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0736225
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0740879
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0744961
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0729548
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728096, upper bound: 0.0731555
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0727585
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0732237
time: 0.32 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.06 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.06
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0729548
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.06
Output dim: 0, lower bound: -0.0728096, upper bound: 0.0731555
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.06
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0727585
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.06
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0732237
Binary search (step 6): status=Status.VERIFIED, low=0.1805223, high=0.1818182, mid=0.1805223, abs_max=0.09181444346904755
rel_dist={0: [-0.07549650593465267, 0.07549650593465268]}

## Binary search (step 7) starts
Candidate diff: 0.1811702


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753536
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753784
time: 0.30 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.77 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.77
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753536
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.77
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753784

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0739373
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
time: 0.28 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753228, upper bound: 0.0734268
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233
time: 0.30 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.56 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.56
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0739373
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.56
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.56
Output dim: 0, lower bound: -0.0753228, upper bound: 0.0734268
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.56
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0751000, upper bound: 0.0733262
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0733124
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961
time: 0.30 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.61 seconds
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.61
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.61
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.61
Output dim: 0, lower bound: -0.0751000, upper bound: 0.0733262
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.61
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0733124
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.61
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.61
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749547
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749518
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749836
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749810
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.32 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0731368
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0731368
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0740920
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749010
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749766
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749765
time: 0.30 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.73 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749547
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749518
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749836
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749810
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.73
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0731368
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.73
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0731368
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0740920
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749010
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749766
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749765

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721131
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722143
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721260
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722157
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724708
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724852
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0733572
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0736225
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0740879
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0744961
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
time: 0.30 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.62 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721131
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722143
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721260
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722157
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724708
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724852
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0733572
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0736225
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0740879
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0744961
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0729548
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0727786
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0727585
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727940, upper bound: 0.0732237
time: 0.33 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.05 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.05
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0729548
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.05
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0727786
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.05
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0727585
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.05
Output dim: 0, lower bound: -0.0727940, upper bound: 0.0732237
Binary search (step 7): status=Status.VERIFIED, low=0.1811702, high=0.1818182, mid=0.1811702, abs_max=0.09181444346904755
rel_dist={0: [-0.07549650593465267, 0.07549650593465268]}

## Binary search (step 8) starts
Candidate diff: 0.1814942


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753536
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753784
time: 0.30 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.77 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.77
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753536
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.77
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753784

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0739373
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
time: 0.29 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753228, upper bound: 0.0734268
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233
time: 0.28 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.56 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.56
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0739373
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.56
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.56
Output dim: 0, lower bound: -0.0753228, upper bound: 0.0734268
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.56
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0733262
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0733124
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961
time: 0.29 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.57 seconds
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.57
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.57
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.57
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0733262
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.57
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0733124
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.57
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.57
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749547
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749518
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749836
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749810
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0740920
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749010
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749766
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749765
time: 0.30 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.74 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749547
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749518
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749836
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749810
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0740920
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749010
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749766
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749765

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721131
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722143
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721260
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722157
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
time: 0.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724708
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724852
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0733572
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0736225
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0740879
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0744961
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
time: 0.30 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.62 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721131
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722143
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721260
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722157
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724708
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724852
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0733572
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0736225
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0740879
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0744961
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0729548
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0731555
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0727585
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0732237
time: 0.29 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.03 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0729548
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0731555
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0727585
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0732237
Binary search (step 8): status=Status.VERIFIED, low=0.1814942, high=0.1818182, mid=0.1814942, abs_max=0.09181444346904755
rel_dist={0: [-0.07549650593465267, 0.07549650593465268]}

## Binary search (step 9) starts
Candidate diff: 0.1816562


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753536
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753784
time: 0.30 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.76 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.76
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753536
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.76
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753784

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0739373
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
time: 0.30 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0734268
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233
time: 0.29 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.59 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.59
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0739373
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.59
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.59
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0734268
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.59
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961
time: 0.27 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.55 seconds
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.55
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.55
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.55
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.55
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749547
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749518
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749836
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749810
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0740920
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749010
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749766
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749765
time: 0.29 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.68 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749547
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749518
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749836
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749810
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0740920
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749010
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749766
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749765

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721131
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722143
time: 0.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721260
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722157
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724708
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724852
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0733572
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0736225
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0740879
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0744961
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
time: 0.30 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.59 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721131
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722143
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721260
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722157
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724708
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724852
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0733572
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0736225
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0740879
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0744961
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0729548
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728096, upper bound: 0.0731555
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0727585
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0732237
time: 0.30 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.00 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.00
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0729548
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.00
Output dim: 0, lower bound: -0.0728096, upper bound: 0.0731555
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.00
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0727585
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.00
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0732237
Binary search (step 9): status=Status.VERIFIED, low=0.1816562, high=0.1818182, mid=0.1816562, abs_max=0.09181444346904755
rel_dist={0: [-0.07549650593465267, 0.07549650593465268]}

## Binary search (step 10) starts
Candidate diff: 0.1817372


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753536
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753784
time: 0.30 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.76 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.76
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753536
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.76
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753784

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0739373
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
time: 0.29 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0734268
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233
time: 0.29 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.52 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.52
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0739373
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.52
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.52
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0734268
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.52
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961
time: 0.28 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.52 seconds
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749547
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749518
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749836
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749810
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0740920
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749010
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749766
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749765
time: 0.29 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.68 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749547
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749518
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749836
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749810
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0740920
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749010
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749766
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749765

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721131
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722143
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721260
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722157
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724708
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724852
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0733572
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0736225
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730059, upper bound: 0.0740879
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0744961
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726716
time: 0.30 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.63 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721131
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722143
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721260
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722157
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724708
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724852
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0733572
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0736225
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0730059, upper bound: 0.0740879
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0744961
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726716

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728027, upper bound: 0.0729548
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728096, upper bound: 0.0731555
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0727585
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727940, upper bound: 0.0732237
time: 0.33 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.06 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.06
Output dim: 0, lower bound: -0.0728027, upper bound: 0.0729548
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.06
Output dim: 0, lower bound: -0.0728096, upper bound: 0.0731555
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.06
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0727585
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.06
Output dim: 0, lower bound: -0.0727940, upper bound: 0.0732237
Binary search (step 10): status=Status.VERIFIED, low=0.1817372, high=0.1818182, mid=0.1817372, abs_max=0.09181444346904755
rel_dist={0: [-0.07549650593465267, 0.07549650593465268]}

## Binary search (step 11) starts
Candidate diff: 0.1817777


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753536
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753784
time: 0.29 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.76 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.76
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753536
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.76
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753784

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0739373
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
time: 0.28 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0734268
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233
time: 0.30 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.59 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.59
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0739373
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.59
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.59
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0734268
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.59
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961
time: 0.28 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.55 seconds
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.55
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.55
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.55
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.55
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749547
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749518
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749836
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749810
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0740920
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749010
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749766
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749765
time: 0.30 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.76 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749547
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749518
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749836
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749810
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0740920
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749010
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749766
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749765

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721131
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722143
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721260
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722157
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724708
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724852
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0733572
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0736225
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0740879
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0744961
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
time: 0.30 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.62 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721131
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722143
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721260
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722157
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724708
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724852
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0733572
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0736225
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0740879
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0744961
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728027, upper bound: 0.0729548
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728096, upper bound: 0.0731555
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0727585
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0732237
time: 0.30 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.04 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.04
Output dim: 0, lower bound: -0.0728027, upper bound: 0.0729548
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.04
Output dim: 0, lower bound: -0.0728096, upper bound: 0.0731555
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.04
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0727585
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.04
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0732237
Binary search (step 11): status=Status.VERIFIED, low=0.1817777, high=0.1818182, mid=0.1817777, abs_max=0.09181444346904755
rel_dist={0: [-0.07549650593465267, 0.07549650593465268]}

## Binary search (step 12) starts
Candidate diff: 0.1817979


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753536
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753784
time: 0.29 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.76 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.76
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753536
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.76
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753784

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0752233, upper bound: 0.0739373
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
time: 0.28 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0734268
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233
time: 0.30 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.58 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.58
Output dim: 0, lower bound: -0.0752233, upper bound: 0.0739373
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.58
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.58
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0734268
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.58
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0736330
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0733262, upper bound: 0.0734262
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961
time: 0.28 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.53 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.53
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0736330
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.53
Output dim: 0, lower bound: -0.0733262, upper bound: 0.0734262
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749547
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749518
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749836
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749810
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0740920
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749010
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.32 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749766
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749765
time: 0.30 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.73 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749547
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749518
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749836
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749810
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0740920
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749010
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749766
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749765

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721131
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722143
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721260
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722157
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724708
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724852
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0733572
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0736225
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0740879
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0744961
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
time: 0.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726716
time: 0.29 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.61 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.61
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721131
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.61
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722143
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.61
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721260
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.61
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722157
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.61
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.61
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.61
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724708
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.61
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724852
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.61
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0733572
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.61
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0736225
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0740879
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0744961
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.61
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.61
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.61
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.61
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726716

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0729548
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728096, upper bound: 0.0731555
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0727585
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727940, upper bound: 0.0732237
time: 0.32 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.06 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.06
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0729548
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.06
Output dim: 0, lower bound: -0.0728096, upper bound: 0.0731555
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.06
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0727585
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.06
Output dim: 0, lower bound: -0.0727940, upper bound: 0.0732237
Binary search (step 12): status=Status.VERIFIED, low=0.1817979, high=0.1818182, mid=0.1817979, abs_max=0.09181444346904755
rel_dist={0: [-0.07549650593465267, 0.07549650593465268]}

## Binary search (step 13) starts
Candidate diff: 0.1818081


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753536
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753784
time: 0.29 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.82 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.82
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753536
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.82
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753784

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0739373
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
time: 0.28 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0734268
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233
time: 0.29 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.58 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.58
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0739373
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.58
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.58
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0734268
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.58
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961
time: 0.29 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.56 seconds
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.56
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.56
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.56
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.56
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749547
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749518
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749836
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749810
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0740920
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749010
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749766
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749765
time: 0.29 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.75 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749547
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749518
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749836
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749810
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0740920
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749010
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749766
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.75
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749765

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721131
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722143
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721260
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722157
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724708
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724852
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0733572
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0736225
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730059, upper bound: 0.0740879
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0744961
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
time: 0.31 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.64 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.64
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721131
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.64
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722143
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.64
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721260
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.64
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722157
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.64
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.64
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.64
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724708
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.64
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724852
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.64
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0733572
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.64
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0736225
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -0.0730059, upper bound: 0.0740879
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0744961
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.64
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.64
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.64
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.64
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728027, upper bound: 0.0729548
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728096, upper bound: 0.0731555
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0727585
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727940, upper bound: 0.0732237
time: 0.33 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.07 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 0, lower bound: -0.0728027, upper bound: 0.0729548
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 0, lower bound: -0.0728096, upper bound: 0.0731555
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0727585
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 0, lower bound: -0.0727940, upper bound: 0.0732237
Binary search (step 13): status=Status.VERIFIED, low=0.1818081, high=0.1818182, mid=0.1818081, abs_max=0.09181444346904755
rel_dist={0: [-0.07549650593465267, 0.07549650593465268]}

## Binary search (step 14) starts
Candidate diff: 0.1818131


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753536
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753784
time: 0.28 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.75 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.75
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753536
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.75
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753784

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0739373
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
time: 0.29 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0734268
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233
time: 0.29 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.61 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.61
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0739373
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.61
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.61
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0734268
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.61
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961
time: 0.28 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.56 seconds
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.56
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.56
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.56
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.56
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749547
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749518
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749836
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749810
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0740920
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749010
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749766
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749765
time: 0.29 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.74 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749547
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749518
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749836
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749810
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0740920
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749010
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749766
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749765

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721131
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722143
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721260
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722157
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724708
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724852
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0733572
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0736225
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0740879
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0744961
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
time: 0.30 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.62 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721131
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722143
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721260
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722157
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724708
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724852
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0733572
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0736225
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0740879
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0744961
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728027, upper bound: 0.0729548
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728096, upper bound: 0.0731555
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0727585
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0732237
time: 0.31 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.02 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.02
Output dim: 0, lower bound: -0.0728027, upper bound: 0.0729548
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.02
Output dim: 0, lower bound: -0.0728096, upper bound: 0.0731555
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.02
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0727585
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.02
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0732237
Binary search (step 14): status=Status.VERIFIED, low=0.1818131, high=0.1818182, mid=0.1818131, abs_max=0.09181444346904755
rel_dist={0: [-0.07549650593465267, 0.07549650593465268]}

## Binary search (step 15) starts
Candidate diff: 0.1818157


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753536
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753784
time: 0.29 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.75 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.75
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753536
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.75
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753784

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0739373
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
time: 0.28 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753228, upper bound: 0.0734268
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233
time: 0.30 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.60 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.60
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0739373
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.60
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.60
Output dim: 0, lower bound: -0.0753228, upper bound: 0.0734268
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.60
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0733262
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0733124
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961
time: 0.28 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.56 seconds
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.56
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.56
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.56
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0733262
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.56
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0733124
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.56
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.56
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749547
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749518
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749836
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749810
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0740920
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749010
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749766
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749765
time: 0.30 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.77 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749547
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749518
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749836
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749810
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0740920
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749010
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749766
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749765

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721131
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722143
time: 0.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721260
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722157
time: 0.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
time: 0.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724708
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724852
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0733572
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0736225
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728441, upper bound: 0.0740879
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0744961
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726716
time: 0.30 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.63 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721131
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722143
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721260
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722157
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724708
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724852
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0733572
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0736225
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0728441, upper bound: 0.0740879
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0744961
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726716

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728027, upper bound: 0.0729548
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728096, upper bound: 0.0731555
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0727585
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727940, upper bound: 0.0732237
time: 0.34 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.06 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.06
Output dim: 0, lower bound: -0.0728027, upper bound: 0.0729548
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.06
Output dim: 0, lower bound: -0.0728096, upper bound: 0.0731555
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.06
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0727585
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.06
Output dim: 0, lower bound: -0.0727940, upper bound: 0.0732237
Binary search (step 15): status=Status.VERIFIED, low=0.1818157, high=0.1818182, mid=0.1818157, abs_max=0.09181444346904755
rel_dist={0: [-0.07549650593465267, 0.07549650593465268]}

## Binary search (step 16) starts
Candidate diff: 0.1818169


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753536
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753784
time: 0.28 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.76 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.76
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753536
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.76
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753784

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0739373
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
time: 0.29 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0734268
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233
time: 0.29 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.57 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.57
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0739373
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.57
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.57
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0734268
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.57
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961
time: 0.28 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.52 seconds
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749547
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749518
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749836
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749810
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0740920
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749010
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749766
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0735295, upper bound: 0.0749765
time: 0.32 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.71 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.71
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749547
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.71
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749518
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.71
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749836
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.71
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749810
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.71
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0740920
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.71
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749010
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.71
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749766
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.71
Output dim: 0, lower bound: -0.0735295, upper bound: 0.0749765

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721131
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722143
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721260
time: 0.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722157
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724708
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724852
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0733572
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0736225
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0740879
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0744961
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
time: 0.30 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.67 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.67
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721131
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.67
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722143
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.67
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721260
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.67
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722157
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.67
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.67
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.67
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724708
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.67
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724852
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.67
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0733572
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.67
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0736225
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0740879
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0744961
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.67
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.67
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.67
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.67
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728027, upper bound: 0.0729548
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728096, upper bound: 0.0731555
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0727585
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0732237
time: 0.31 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.12 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.12
Output dim: 0, lower bound: -0.0728027, upper bound: 0.0729548
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.12
Output dim: 0, lower bound: -0.0728096, upper bound: 0.0731555
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.12
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0727585
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.12
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0732237
Binary search (step 16): status=Status.VERIFIED, low=0.1818169, high=0.1818182, mid=0.1818169, abs_max=0.09181444346904755
rel_dist={0: [-0.07549650593465267, 0.07549650593465268]}

## Binary search (step 17) starts
Candidate diff: 0.1818176


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753536
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753784
time: 0.28 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.76 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.76
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753536
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.76
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753784

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0739373
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
time: 0.28 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0734268
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233
time: 0.30 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.62 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.62
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0739373
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.62
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.62
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0734268
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.62
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961
time: 0.29 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.60 seconds
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.60
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.60
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.60
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.60
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749547
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749518
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749836
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749810
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0740920
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749010
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749766
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749765
time: 0.29 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.72 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.72
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749547
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.72
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749518
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.72
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749836
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.72
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749810
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.72
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0740920
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.72
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749010
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.72
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749766
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.72
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749765

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721131
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722143
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721260
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722157
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724708
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724852
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0733572
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0736225
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0740879
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0744961
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
time: 0.30 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.64 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.64
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721131
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.64
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722143
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.64
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721260
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.64
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722157
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.64
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.64
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724923
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.64
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724708
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.64
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0724852
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.64
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0733572
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.64
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0736225
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0740879
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.64
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0744961
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.64
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.64
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0726352
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.64
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.64
Output dim: 0, lower bound: -0.0721108, upper bound: 0.0726716

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0729548
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0731555
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0727585
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0732237
time: 0.29 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.05 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.05
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0729548
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.05
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0731555
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.05
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0727585
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.05
Output dim: 0, lower bound: -0.0727585, upper bound: 0.0732237
Binary search (step 17): status=Status.VERIFIED, low=0.1818176, high=0.1818182, mid=0.1818176, abs_max=0.09181444346904755
rel_dist={0: [-0.07549650593465267, 0.07549650593465268]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.18181755448245165
execution time: 933.56 seconds
