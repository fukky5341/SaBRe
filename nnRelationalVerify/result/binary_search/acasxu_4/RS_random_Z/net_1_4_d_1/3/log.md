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
execution time: IAR + LP analysis = 1.90 + 0.87 = 2.77 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0754965, upper bound: 0.0754965


# Binary Search by BASE starts (time budget: 1197.23 seconds, max iter: 100)

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
Binary search time: 49.40 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01594543504552348


# Relational Split (RS_random_Z) starts
Time budget: 1147.83 seconds

## Binary search (step 0) starts
Candidate diff: 0.0988818


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0740421, upper bound: 0.0740421
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0740421, upper bound: 0.0754287
time: 0.29 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.60 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.60
Output dim: 0, lower bound: -0.0740421, upper bound: 0.0740421
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.60
Output dim: 0, lower bound: -0.0740421, upper bound: 0.0754287

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0739197
time: 0.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0734268
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
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0751693
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0751743
time: 0.28 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.36 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.36
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0739197
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.36
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0734268
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.36
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0751693
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.36
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0751743

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
time: 0.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961
time: 0.29 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.39 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0732624, upper bound: 0.0737588
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0732624, upper bound: 0.0749328
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0732624, upper bound: 0.0736986
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0732624, upper bound: 0.0749004
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0722488, upper bound: 0.0725979
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0722400, upper bound: 0.0725979
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0732285, upper bound: 0.0750137
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0732285, upper bound: 0.0744276
time: 0.30 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.38 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.38
Output dim: 0, lower bound: -0.0732624, upper bound: 0.0737588
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -0.0732624, upper bound: 0.0749328
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.38
Output dim: 0, lower bound: -0.0732624, upper bound: 0.0736986
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -0.0732624, upper bound: 0.0749004
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.38
Output dim: 0, lower bound: -0.0722488, upper bound: 0.0725979
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.38
Output dim: 0, lower bound: -0.0722400, upper bound: 0.0725979
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -0.0732285, upper bound: 0.0750137
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -0.0732285, upper bound: 0.0744276

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730311, upper bound: 0.0747647
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0730311, upper bound: 0.0730344
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0729715, upper bound: 0.0730335
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0729715, upper bound: 0.0730335
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
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0730698, upper bound: 0.0731598
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730698, upper bound: 0.0748767
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730559, upper bound: 0.0742426
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730559, upper bound: 0.0742603
time: 0.31 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.41 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.41
Output dim: 0, lower bound: -0.0730311, upper bound: 0.0747647
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.41
Output dim: 0, lower bound: -0.0730311, upper bound: 0.0730344
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.41
Output dim: 0, lower bound: -0.0729715, upper bound: 0.0730335
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.41
Output dim: 0, lower bound: -0.0729715, upper bound: 0.0730335
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.41
Output dim: 0, lower bound: -0.0730698, upper bound: 0.0731598
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.41
Output dim: 0, lower bound: -0.0730698, upper bound: 0.0748767
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.41
Output dim: 0, lower bound: -0.0730559, upper bound: 0.0742426
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.41
Output dim: 0, lower bound: -0.0730559, upper bound: 0.0742603

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728536, upper bound: 0.0746950
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728536, upper bound: 0.0746940
time: 0.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0718765, upper bound: 0.0724738
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0718765, upper bound: 0.0724738
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0730027, upper bound: 0.0730027
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730027, upper bound: 0.0742016
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728347, upper bound: 0.0733866
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0732270, upper bound: 0.0728347
time: 0.32 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.88 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0728536, upper bound: 0.0746950
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0728536, upper bound: 0.0746940
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0718765, upper bound: 0.0724738
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0718765, upper bound: 0.0724738
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0730027, upper bound: 0.0730027
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0730027, upper bound: 0.0742016
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0728347, upper bound: 0.0733866
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0732270, upper bound: 0.0728347

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0727709, upper bound: 0.0746157
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727709, upper bound: 0.0728158
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 15

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0718351, upper bound: 0.0718351
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0718351, upper bound: 0.0720474
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0719954, upper bound: 0.0724162
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0719954, upper bound: 0.0724162
time: 0.30 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.44 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0727709, upper bound: 0.0746157
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0727709, upper bound: 0.0728158
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0718351, upper bound: 0.0718351
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0718351, upper bound: 0.0720474
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0719954, upper bound: 0.0724162
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0719954, upper bound: 0.0724162

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0726028, upper bound: 0.0738872
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0726028, upper bound: 0.0744183
time: 0.30 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.42 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.42
Output dim: 0, lower bound: -0.0726028, upper bound: 0.0738872
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.42
Output dim: 0, lower bound: -0.0726028, upper bound: 0.0744183

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0722366, upper bound: 0.0735704
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0722366, upper bound: 0.0737559
time: 0.32 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 2.53 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.53
Output dim: 0, lower bound: -0.0722366, upper bound: 0.0735704
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.53
Output dim: 0, lower bound: -0.0722366, upper bound: 0.0737559
Binary search (step 0): status=Status.VERIFIED, low=0.0988818, high=0.1818182, mid=0.0988818, abs_max=0.09181444346904755
rel_dist={0: [-0.07549650593465267, 0.07549650593465268]}

## Binary search (step 1) starts
Candidate diff: 0.1403500


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0754287, upper bound: 0.0740940
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0740940, upper bound: 0.0754287
time: 0.32 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.66 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.66
Output dim: 0, lower bound: -0.0754287, upper bound: 0.0740940
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.66
Output dim: 0, lower bound: -0.0740940, upper bound: 0.0754287

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0751743, upper bound: 0.0737033
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0751693, upper bound: 0.0734733
time: 0.30 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233
time: 0.32 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.66 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.66
Output dim: 0, lower bound: -0.0751743, upper bound: 0.0737033
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.66
Output dim: 0, lower bound: -0.0751693, upper bound: 0.0734733
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.66
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.66
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0750961, upper bound: 0.0736330
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0733262
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0750934, upper bound: 0.0734262
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0733124
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

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
- Time for RS candidates: 2.51 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.51
Output dim: 0, lower bound: -0.0750961, upper bound: 0.0736330
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.51
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0733262
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.51
Output dim: 0, lower bound: -0.0750934, upper bound: 0.0734262
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.51
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0733124
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.51
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.51
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.51
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.51
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0735295
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0749766, upper bound: 0.0734823
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0724417, upper bound: 0.0722400
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0723976, upper bound: 0.0722400
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0722400, upper bound: 0.0722848
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0722400, upper bound: 0.0724194
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0731438, upper bound: 0.0732505
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731438, upper bound: 0.0749599
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0731094, upper bound: 0.0735058
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0731114, upper bound: 0.0735532
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
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0732624, upper bound: 0.0744456
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733071, upper bound: 0.0749951
time: 0.30 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.44 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0735295
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0749766, upper bound: 0.0734823
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0724417, upper bound: 0.0722400
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0723976, upper bound: 0.0722400
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0722400, upper bound: 0.0722848
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0722400, upper bound: 0.0724194
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0731438, upper bound: 0.0732505
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0731438, upper bound: 0.0749599
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0731094, upper bound: 0.0735058
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0731114, upper bound: 0.0735532
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0732624, upper bound: 0.0744456
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0733071, upper bound: 0.0749951

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0726352, upper bound: 0.0721101
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0726352, upper bound: 0.0721101
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0729209, upper bound: 0.0747312
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0729209, upper bound: 0.0729209
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
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0729715, upper bound: 0.0730227
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0729715, upper bound: 0.0730227
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0731158, upper bound: 0.0731671
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730901, upper bound: 0.0748546
time: 0.29 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.43 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.43
Output dim: 0, lower bound: -0.0726352, upper bound: 0.0721101
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.43
Output dim: 0, lower bound: -0.0726352, upper bound: 0.0721101
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -0.0729209, upper bound: 0.0747312
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.43
Output dim: 0, lower bound: -0.0729209, upper bound: 0.0729209
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.43
Output dim: 0, lower bound: -0.0729715, upper bound: 0.0730227
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.43
Output dim: 0, lower bound: -0.0729715, upper bound: 0.0730227
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.43
Output dim: 0, lower bound: -0.0731158, upper bound: 0.0731671
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -0.0730901, upper bound: 0.0748546

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0726267, upper bound: 0.0729128
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0726461, upper bound: 0.0729128
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 39

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727860, upper bound: 0.0730757
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0727860, upper bound: 0.0745927
time: 0.30 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.41 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.41
Output dim: 0, lower bound: -0.0726267, upper bound: 0.0729128
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.41
Output dim: 0, lower bound: -0.0726461, upper bound: 0.0729128
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.41
Output dim: 0, lower bound: -0.0727860, upper bound: 0.0730757
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -0.0727860, upper bound: 0.0745927

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 39

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0727232, upper bound: 0.0745246
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0727232, upper bound: 0.0741191
time: 0.32 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.90 seconds
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.90
Output dim: 0, lower bound: -0.0727232, upper bound: 0.0745246
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.90
Output dim: 0, lower bound: -0.0727232, upper bound: 0.0741191

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0724255, upper bound: 0.0741692
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0724255, upper bound: 0.0724255
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0725391, upper bound: 0.0738607
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0725391, upper bound: 0.0739248
time: 0.30 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.85 seconds
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.85
Output dim: 0, lower bound: -0.0724255, upper bound: 0.0741692
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.85
Output dim: 0, lower bound: -0.0724255, upper bound: 0.0724255
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.85
Output dim: 0, lower bound: -0.0725391, upper bound: 0.0738607
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.85
Output dim: 0, lower bound: -0.0725391, upper bound: 0.0739248

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 39

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0722970, upper bound: 0.0722970
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0722970, upper bound: 0.0724722
time: 0.32 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 3.42 seconds
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.42
Output dim: 0, lower bound: -0.0722970, upper bound: 0.0722970
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.42
Output dim: 0, lower bound: -0.0722970, upper bound: 0.0724722
Binary search (step 1): status=Status.VERIFIED, low=0.1403500, high=0.1818182, mid=0.1403500, abs_max=0.09181444346904755
rel_dist={0: [-0.07549650593465267, 0.07549650593465268]}

## Binary search (step 2) starts
Candidate diff: 0.1610841


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0751693, upper bound: 0.0751693
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0751693, upper bound: 0.0751743
time: 0.29 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.61 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.61
Output dim: 0, lower bound: -0.0751693, upper bound: 0.0751693
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.61
Output dim: 0, lower bound: -0.0751693, upper bound: 0.0751743

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
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0737033
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0751693
time: 0.30 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0750927, upper bound: 0.0751000
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0750927, upper bound: 0.0750980
time: 0.30 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.35 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.35
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0737033
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.35
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0751693
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.35
Output dim: 0, lower bound: -0.0750927, upper bound: 0.0751000
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.35
Output dim: 0, lower bound: -0.0750927, upper bound: 0.0750980

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0734262
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
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

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0733124
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961
time: 0.29 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.41 seconds
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.41
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0734262
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.41
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0733124
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730145, upper bound: 0.0745164
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730145, upper bound: 0.0747252
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0732285, upper bound: 0.0750105
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0732285, upper bound: 0.0737973
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0732285, upper bound: 0.0750210
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0732285, upper bound: 0.0733270
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730980, upper bound: 0.0748469
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0731448, upper bound: 0.0730980
time: 0.31 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.44 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0730145, upper bound: 0.0745164
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0730145, upper bound: 0.0747252
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0732285, upper bound: 0.0750105
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0732285, upper bound: 0.0737973
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0732285, upper bound: 0.0750210
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0732285, upper bound: 0.0733270
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0730980, upper bound: 0.0748469
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0731448, upper bound: 0.0730980

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727703, upper bound: 0.0727703
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727703, upper bound: 0.0727736
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728619, upper bound: 0.0732268
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728619, upper bound: 0.0746200
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0729940, upper bound: 0.0730259
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0729940, upper bound: 0.0730259
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0730487, upper bound: 0.0735330
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0730492, upper bound: 0.0735384
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727890, upper bound: 0.0727890
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727890, upper bound: 0.0729773
time: 0.31 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.46 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.46
Output dim: 0, lower bound: -0.0727703, upper bound: 0.0727703
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.46
Output dim: 0, lower bound: -0.0727703, upper bound: 0.0727736
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.46
Output dim: 0, lower bound: -0.0728619, upper bound: 0.0732268
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.46
Output dim: 0, lower bound: -0.0728619, upper bound: 0.0746200
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.46
Output dim: 0, lower bound: -0.0729940, upper bound: 0.0730259
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.46
Output dim: 0, lower bound: -0.0729940, upper bound: 0.0730259
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.46
Output dim: 0, lower bound: -0.0730487, upper bound: 0.0735330
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.46
Output dim: 0, lower bound: -0.0730492, upper bound: 0.0735384
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.46
Output dim: 0, lower bound: -0.0727890, upper bound: 0.0727890
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.46
Output dim: 0, lower bound: -0.0727890, upper bound: 0.0729773

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0726935, upper bound: 0.0744150
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0726935, upper bound: 0.0744264
time: 0.30 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.50 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 0, lower bound: -0.0726935, upper bound: 0.0744150
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.50
Output dim: 0, lower bound: -0.0726935, upper bound: 0.0744264

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0724021, upper bound: 0.0741611
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0724021, upper bound: 0.0724021
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0724388, upper bound: 0.0724388
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0724388, upper bound: 0.0724533
time: 0.32 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.49 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.49
Output dim: 0, lower bound: -0.0724021, upper bound: 0.0741611
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.49
Output dim: 0, lower bound: -0.0724021, upper bound: 0.0724021
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.49
Output dim: 0, lower bound: -0.0724388, upper bound: 0.0724388
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.49
Output dim: 0, lower bound: -0.0724388, upper bound: 0.0724533

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0723359, upper bound: 0.0740846
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0723359, upper bound: 0.0723359
time: 0.31 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.52 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -0.0723359, upper bound: 0.0740846
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.52
Output dim: 0, lower bound: -0.0723359, upper bound: 0.0723359

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Candidate
type: RSZ, layer: 3, pos: 15

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0722444, upper bound: 0.0725711
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0722444, upper bound: 0.0725711
time: 0.31 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 2.50 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.50
Output dim: 0, lower bound: -0.0722444, upper bound: 0.0725711
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.50
Output dim: 0, lower bound: -0.0722444, upper bound: 0.0725711
Binary search (step 2): status=Status.VERIFIED, low=0.1610841, high=0.1818182, mid=0.1610841, abs_max=0.09181444346904755
rel_dist={0: [-0.07549650593465267, 0.07549650593465268]}

## Binary search (step 3) starts
Candidate diff: 0.1714511


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0751693, upper bound: 0.0751693
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0751693, upper bound: 0.0751743
time: 0.28 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.60 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.60
Output dim: 0, lower bound: -0.0751693, upper bound: 0.0751693
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.60
Output dim: 0, lower bound: -0.0751693, upper bound: 0.0751743

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
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0737033
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0751693
time: 0.31 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0750934, upper bound: 0.0751000
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0750927, upper bound: 0.0750980
time: 0.30 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.36 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.36
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0737033
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.36
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0751693
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.36
Output dim: 0, lower bound: -0.0750934, upper bound: 0.0751000
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.36
Output dim: 0, lower bound: -0.0750927, upper bound: 0.0750980

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0734262
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
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
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0733124
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0736330, upper bound: 0.0750961
time: 0.31 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.45 seconds
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.45
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.45
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.45
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0734262
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.45
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.45
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0733124
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.45
Output dim: 0, lower bound: -0.0736330, upper bound: 0.0750961

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731438, upper bound: 0.0744930
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731438, upper bound: 0.0749422
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730145, upper bound: 0.0743565
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730145, upper bound: 0.0747076
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0730615, upper bound: 0.0730615
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0730615, upper bound: 0.0730615
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731010, upper bound: 0.0748469
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0733882, upper bound: 0.0730980
time: 0.32 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.44 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0731438, upper bound: 0.0744930
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0731438, upper bound: 0.0749422
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0730145, upper bound: 0.0743565
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0730145, upper bound: 0.0747076
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0730615, upper bound: 0.0730615
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0730615, upper bound: 0.0730615
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0731010, upper bound: 0.0748469
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0733882, upper bound: 0.0730980

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728619, upper bound: 0.0732984
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728619, upper bound: 0.0732268
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0729835, upper bound: 0.0748043
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0729835, upper bound: 0.0748030
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728619, upper bound: 0.0732265
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728619, upper bound: 0.0742530
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0729236, upper bound: 0.0729277
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0729236, upper bound: 0.0733168
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
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0729276, upper bound: 0.0731081
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0729556, upper bound: 0.0746862
time: 0.32 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.50 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.50
Output dim: 0, lower bound: -0.0728619, upper bound: 0.0732984
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.50
Output dim: 0, lower bound: -0.0728619, upper bound: 0.0732268
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.50
Output dim: 0, lower bound: -0.0729835, upper bound: 0.0748043
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.50
Output dim: 0, lower bound: -0.0729835, upper bound: 0.0748030
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.50
Output dim: 0, lower bound: -0.0728619, upper bound: 0.0732265
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.50
Output dim: 0, lower bound: -0.0728619, upper bound: 0.0742530
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.50
Output dim: 0, lower bound: -0.0729236, upper bound: 0.0729277
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.50
Output dim: 0, lower bound: -0.0729236, upper bound: 0.0733168
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.50
Output dim: 0, lower bound: -0.0729276, upper bound: 0.0731081
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.50
Output dim: 0, lower bound: -0.0729556, upper bound: 0.0746862

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728139, upper bound: 0.0734813
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728139, upper bound: 0.0734844
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0726935, upper bound: 0.0741926
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0726935, upper bound: 0.0744264
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0726935, upper bound: 0.0732678
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0726935, upper bound: 0.0739826
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0727501, upper bound: 0.0745476
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0727501, upper bound: 0.0745337
time: 0.30 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.44 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.44
Output dim: 0, lower bound: -0.0728139, upper bound: 0.0734813
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.44
Output dim: 0, lower bound: -0.0728139, upper bound: 0.0734844
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.44
Output dim: 0, lower bound: -0.0726935, upper bound: 0.0741926
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.44
Output dim: 0, lower bound: -0.0726935, upper bound: 0.0744264
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.44
Output dim: 0, lower bound: -0.0726935, upper bound: 0.0732678
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.44
Output dim: 0, lower bound: -0.0726935, upper bound: 0.0739826
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.44
Output dim: 0, lower bound: -0.0727501, upper bound: 0.0745476
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.44
Output dim: 0, lower bound: -0.0727501, upper bound: 0.0745337

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0724021, upper bound: 0.0740189
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0724021, upper bound: 0.0724021
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0726092, upper bound: 0.0730803
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0726092, upper bound: 0.0731922
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0726677, upper bound: 0.0744712
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0726677, upper bound: 0.0739544
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 15

### Candidate
type: RSZ, layer: 3, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0726849, upper bound: 0.0740060
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0726849, upper bound: 0.0744547
time: 0.32 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.48 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0724021, upper bound: 0.0740189
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0724021, upper bound: 0.0724021
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0726092, upper bound: 0.0730803
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0726092, upper bound: 0.0731922
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0726677, upper bound: 0.0744712
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0726677, upper bound: 0.0739544
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0726849, upper bound: 0.0740060
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0726849, upper bound: 0.0744547

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0723028, upper bound: 0.0725855
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0723028, upper bound: 0.0725866
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Candidate
type: RSZ, layer: 3, pos: 26

### Candidate
type: RSZ, layer: 3, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0726028, upper bound: 0.0739400
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0726028, upper bound: 0.0744056
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0723034, upper bound: 0.0723034
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0723034, upper bound: 0.0738061
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0724039, upper bound: 0.0724039
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0724039, upper bound: 0.0726679
time: 0.31 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.48 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 0, lower bound: -0.0723028, upper bound: 0.0725855
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 0, lower bound: -0.0723028, upper bound: 0.0725866
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 0, lower bound: -0.0726028, upper bound: 0.0739400
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.48
Output dim: 0, lower bound: -0.0726028, upper bound: 0.0744056
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 0, lower bound: -0.0723034, upper bound: 0.0723034
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 0, lower bound: -0.0723034, upper bound: 0.0738061
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 0, lower bound: -0.0724039, upper bound: 0.0724039
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.48
Output dim: 0, lower bound: -0.0724039, upper bound: 0.0726679

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0723426, upper bound: 0.0723426
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0723426, upper bound: 0.0725496
time: 0.31 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 2.46 seconds
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.46
Output dim: 0, lower bound: -0.0723426, upper bound: 0.0723426
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.46
Output dim: 0, lower bound: -0.0723426, upper bound: 0.0725496
Binary search (step 3): status=Status.VERIFIED, low=0.1714511, high=0.1818182, mid=0.1714511, abs_max=0.09181444346904755
rel_dist={0: [-0.07549650593465267, 0.07549650593465268]}

## Binary search (step 4) starts
Candidate diff: 0.1766347


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0741131, upper bound: 0.0741131
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0741131, upper bound: 0.0754287
time: 0.30 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.61 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.61
Output dim: 0, lower bound: -0.0741131, upper bound: 0.0741131
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.61
Output dim: 0, lower bound: -0.0741131, upper bound: 0.0754287

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0739373
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0734268
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
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0751693
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0751743
time: 0.30 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.40 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.40
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0739373
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.40
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0734268
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.40
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0751693
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.40
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0751743

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
time: 0.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961
time: 0.28 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.38 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.38
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.38
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.38
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.38
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0730615, upper bound: 0.0730888
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0730615, upper bound: 0.0730888
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0722400, upper bound: 0.0723976
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0722400, upper bound: 0.0724417
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0732285, upper bound: 0.0750152
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0732285, upper bound: 0.0733270
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0732285, upper bound: 0.0750137
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0732285, upper bound: 0.0744276
time: 0.29 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.42 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0730615, upper bound: 0.0730888
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0730615, upper bound: 0.0730888
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0722400, upper bound: 0.0723976
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0722400, upper bound: 0.0724417
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0732285, upper bound: 0.0750152
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0732285, upper bound: 0.0733270
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0732285, upper bound: 0.0750137
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0732285, upper bound: 0.0744276

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731787, upper bound: 0.0743582
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731787, upper bound: 0.0749161
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721675, upper bound: 0.0728004
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721675, upper bound: 0.0728004
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730508, upper bound: 0.0742570
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0733275, upper bound: 0.0730160
time: 0.31 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.51 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.51
Output dim: 0, lower bound: -0.0731787, upper bound: 0.0743582
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.51
Output dim: 0, lower bound: -0.0731787, upper bound: 0.0749161
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 0, lower bound: -0.0721675, upper bound: 0.0728004
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 0, lower bound: -0.0721675, upper bound: 0.0728004
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.51
Output dim: 0, lower bound: -0.0730508, upper bound: 0.0742570
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 0, lower bound: -0.0733275, upper bound: 0.0730160

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0730136, upper bound: 0.0731431
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730136, upper bound: 0.0742136
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730027, upper bound: 0.0748465
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730027, upper bound: 0.0748468
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0729546, upper bound: 0.0741425
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0729546, upper bound: 0.0742153
time: 0.29 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.53 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.53
Output dim: 0, lower bound: -0.0730136, upper bound: 0.0731431
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -0.0730136, upper bound: 0.0742136
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -0.0730027, upper bound: 0.0748465
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -0.0730027, upper bound: 0.0748468
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -0.0729546, upper bound: 0.0741425
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.53
Output dim: 0, lower bound: -0.0729546, upper bound: 0.0742153

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727874, upper bound: 0.0731006
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727975, upper bound: 0.0731128
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 48

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0726751, upper bound: 0.0735710
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0726751, upper bound: 0.0745393
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0726751, upper bound: 0.0726751
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0726751, upper bound: 0.0744187
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 15

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727795, upper bound: 0.0729350
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727795, upper bound: 0.0739717
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0718639, upper bound: 0.0721453
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0718639, upper bound: 0.0726835
time: 0.32 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.53 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0727874, upper bound: 0.0731006
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0727975, upper bound: 0.0731128
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0726751, upper bound: 0.0735710
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0726751, upper bound: 0.0745393
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0726751, upper bound: 0.0726751
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0726751, upper bound: 0.0744187
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0727795, upper bound: 0.0729350
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0727795, upper bound: 0.0739717
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0718639, upper bound: 0.0721453
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0718639, upper bound: 0.0726835

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0725391, upper bound: 0.0725391
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0725391, upper bound: 0.0744443
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 48

### Candidate
type: RSZ, layer: 3, pos: 26

### Candidate
type: RSZ, layer: 3, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0723792, upper bound: 0.0741288
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0723792, upper bound: 0.0723792
time: 0.31 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 3.00 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.00
Output dim: 0, lower bound: -0.0725391, upper bound: 0.0725391
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.00
Output dim: 0, lower bound: -0.0725391, upper bound: 0.0744443
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.00
Output dim: 0, lower bound: -0.0723792, upper bound: 0.0741288
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.00
Output dim: 0, lower bound: -0.0723792, upper bound: 0.0723792

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 39

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0722802, upper bound: 0.0722802
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0722802, upper bound: 0.0723188
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Candidate
type: RSZ, layer: 3, pos: 39

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 48

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0722366, upper bound: 0.0722366
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0722366, upper bound: 0.0739907
time: 0.31 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 3.97 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.97
Output dim: 0, lower bound: -0.0722802, upper bound: 0.0722802
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.97
Output dim: 0, lower bound: -0.0722802, upper bound: 0.0723188
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.97
Output dim: 0, lower bound: -0.0722366, upper bound: 0.0722366
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.97
Output dim: 0, lower bound: -0.0722366, upper bound: 0.0739907

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 26

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 48

### Candidate
type: RSZ, layer: 3, pos: 7

### Candidate
type: RSZ, layer: 3, pos: 39

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721416, upper bound: 0.0723534
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721416, upper bound: 0.0724055
time: 0.31 seconds

## Summary of splitting (split count: 9)
- Time for RS candidates: 3.98 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 3.98
Output dim: 0, lower bound: -0.0721416, upper bound: 0.0723534
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 3.98
Output dim: 0, lower bound: -0.0721416, upper bound: 0.0724055
Binary search (step 4): status=Status.VERIFIED, low=0.1766347, high=0.1818182, mid=0.1766347, abs_max=0.09181444346904755
rel_dist={0: [-0.07549650593465267, 0.07549650593465268]}

## Binary search (step 5) starts
Candidate diff: 0.1792264


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0741131, upper bound: 0.0741131
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0741131, upper bound: 0.0754287
time: 0.30 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.62 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.62
Output dim: 0, lower bound: -0.0741131, upper bound: 0.0741131
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.62
Output dim: 0, lower bound: -0.0741131, upper bound: 0.0754287

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0737033
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0734733
time: 0.32 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233
time: 0.30 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.46 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.46
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0737033
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.46
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0734733
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.46
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.46
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
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

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

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
- Time for RS candidates: 2.45 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.45
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.45
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.45
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.45
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730980, upper bound: 0.0748982
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0730980, upper bound: 0.0730998
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0722488, upper bound: 0.0725979
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0722488, upper bound: 0.0725979
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0732436, upper bound: 0.0738485
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731438, upper bound: 0.0749413
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730980, upper bound: 0.0748469
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0731448, upper bound: 0.0730980
time: 0.31 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.53 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.53
Output dim: 0, lower bound: -0.0730980, upper bound: 0.0748982
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.53
Output dim: 0, lower bound: -0.0730980, upper bound: 0.0730998
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.53
Output dim: 0, lower bound: -0.0722488, upper bound: 0.0725979
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.53
Output dim: 0, lower bound: -0.0722488, upper bound: 0.0725979
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.53
Output dim: 0, lower bound: -0.0732436, upper bound: 0.0738485
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.53
Output dim: 0, lower bound: -0.0731438, upper bound: 0.0749413
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.53
Output dim: 0, lower bound: -0.0730980, upper bound: 0.0748469
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.53
Output dim: 0, lower bound: -0.0731448, upper bound: 0.0730980

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0729209, upper bound: 0.0743966
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0729209, upper bound: 0.0747275
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0730901, upper bound: 0.0735312
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730901, upper bound: 0.0747658
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730160, upper bound: 0.0747640
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730160, upper bound: 0.0742570
time: 0.32 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.68 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 0, lower bound: -0.0729209, upper bound: 0.0743966
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 0, lower bound: -0.0729209, upper bound: 0.0747275
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.68
Output dim: 0, lower bound: -0.0730901, upper bound: 0.0735312
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 0, lower bound: -0.0730901, upper bound: 0.0747658
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 0, lower bound: -0.0730160, upper bound: 0.0747640
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 0, lower bound: -0.0730160, upper bound: 0.0742570

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0717156, upper bound: 0.0717417
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0717156, upper bound: 0.0717795
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0726267, upper bound: 0.0728965
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0726267, upper bound: 0.0728965
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730136, upper bound: 0.0746772
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0730136, upper bound: 0.0735988
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0726681, upper bound: 0.0730870
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0726681, upper bound: 0.0744521
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0719119, upper bound: 0.0727019
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0719119, upper bound: 0.0727019
time: 0.31 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.17 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.17
Output dim: 0, lower bound: -0.0717156, upper bound: 0.0717417
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.17
Output dim: 0, lower bound: -0.0717156, upper bound: 0.0717795
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.17
Output dim: 0, lower bound: -0.0726267, upper bound: 0.0728965
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.17
Output dim: 0, lower bound: -0.0726267, upper bound: 0.0728965
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 0, lower bound: -0.0730136, upper bound: 0.0746772
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.17
Output dim: 0, lower bound: -0.0730136, upper bound: 0.0735988
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.17
Output dim: 0, lower bound: -0.0726681, upper bound: 0.0730870
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.17
Output dim: 0, lower bound: -0.0726681, upper bound: 0.0744521
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.17
Output dim: 0, lower bound: -0.0719119, upper bound: 0.0727019
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.17
Output dim: 0, lower bound: -0.0719119, upper bound: 0.0727019

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728480, upper bound: 0.0737321
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728480, upper bound: 0.0745167
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0725595, upper bound: 0.0725595
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0725595, upper bound: 0.0727232
time: 0.32 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.71 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 0, lower bound: -0.0728480, upper bound: 0.0737321
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -0.0728480, upper bound: 0.0745167
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 0, lower bound: -0.0725595, upper bound: 0.0725595
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 0, lower bound: -0.0725595, upper bound: 0.0727232

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0726028, upper bound: 0.0742604
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0726028, upper bound: 0.0726028
time: 0.33 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 3.24 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 0, lower bound: -0.0726028, upper bound: 0.0742604
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.24
Output dim: 0, lower bound: -0.0726028, upper bound: 0.0726028

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 49

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0722366, upper bound: 0.0732810
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0722366, upper bound: 0.0736379
time: 0.32 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 3.23 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.23
Output dim: 0, lower bound: -0.0722366, upper bound: 0.0732810
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.23
Output dim: 0, lower bound: -0.0722366, upper bound: 0.0736379
Binary search (step 5): status=Status.VERIFIED, low=0.1792264, high=0.1818182, mid=0.1792264, abs_max=0.09181444346904755
rel_dist={0: [-0.07549650593465267, 0.07549650593465268]}

## Binary search (step 6) starts
Candidate diff: 0.1805223


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0741131, upper bound: 0.0741131
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0741131, upper bound: 0.0754287
time: 0.31 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.65 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.65
Output dim: 0, lower bound: -0.0741131, upper bound: 0.0741131
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.65
Output dim: 0, lower bound: -0.0741131, upper bound: 0.0754287

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0737033
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0734733
time: 0.34 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0751693
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0751743
time: 0.31 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.67 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.67
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0737033
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.67
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0734733
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.67
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0751693
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.67
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0751743

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961
time: 0.30 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.67 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.67
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.67
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.67
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.67
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731438, upper bound: 0.0744930
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731438, upper bound: 0.0749422
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0732436, upper bound: 0.0738485
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731438, upper bound: 0.0749413
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0732624, upper bound: 0.0737588
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0732624, upper bound: 0.0749948
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

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
- Time for RS candidates: 2.68 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0731438, upper bound: 0.0744930
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0731438, upper bound: 0.0749422
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0732436, upper bound: 0.0738485
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0731438, upper bound: 0.0749413
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0732624, upper bound: 0.0737588
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0732624, upper bound: 0.0749948
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749766
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749765

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0729527, upper bound: 0.0733936
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0729527, upper bound: 0.0733936
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0729209, upper bound: 0.0747275
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0729209, upper bound: 0.0729209
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0729527, upper bound: 0.0733578
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0729527, upper bound: 0.0734285
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
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731787, upper bound: 0.0749161
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0731787, upper bound: 0.0732653
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730559, upper bound: 0.0748939
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730559, upper bound: 0.0742426
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730860, upper bound: 0.0743317
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730860, upper bound: 0.0749203
time: 0.32 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.00 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.00
Output dim: 0, lower bound: -0.0729527, upper bound: 0.0733936
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.00
Output dim: 0, lower bound: -0.0729527, upper bound: 0.0733936
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 0, lower bound: -0.0729209, upper bound: 0.0747275
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.00
Output dim: 0, lower bound: -0.0729209, upper bound: 0.0729209
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.00
Output dim: 0, lower bound: -0.0729527, upper bound: 0.0733578
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.00
Output dim: 0, lower bound: -0.0729527, upper bound: 0.0734285
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 0, lower bound: -0.0731787, upper bound: 0.0749161
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.00
Output dim: 0, lower bound: -0.0731787, upper bound: 0.0732653
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 0, lower bound: -0.0730559, upper bound: 0.0748939
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 0, lower bound: -0.0730559, upper bound: 0.0742426
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 0, lower bound: -0.0730860, upper bound: 0.0743317
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 0, lower bound: -0.0730860, upper bound: 0.0749203

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728400, upper bound: 0.0746479
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728400, upper bound: 0.0729531
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0729086, upper bound: 0.0729086
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0729086, upper bound: 0.0729086
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727945, upper bound: 0.0727945
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727945, upper bound: 0.0727945
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728898, upper bound: 0.0728898
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728918, upper bound: 0.0729235
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728536, upper bound: 0.0741593
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0731669, upper bound: 0.0728536
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728536, upper bound: 0.0746330
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0729096, upper bound: 0.0728536
time: 0.32 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.60 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0728400, upper bound: 0.0746479
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0728400, upper bound: 0.0729531
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0729086, upper bound: 0.0729086
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0729086, upper bound: 0.0729086
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0727945, upper bound: 0.0727945
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0727945, upper bound: 0.0727945
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0728898, upper bound: 0.0728898
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0728918, upper bound: 0.0729235
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0728536, upper bound: 0.0741593
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0731669, upper bound: 0.0728536
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0728536, upper bound: 0.0746330
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0729096, upper bound: 0.0728536

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0726677, upper bound: 0.0745039
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0726677, upper bound: 0.0744779
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0727709, upper bound: 0.0740810
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0727709, upper bound: 0.0740323
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0724557, upper bound: 0.0726062
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0724557, upper bound: 0.0742264
time: 0.32 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.57 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -0.0726677, upper bound: 0.0745039
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -0.0726677, upper bound: 0.0744779
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -0.0727709, upper bound: 0.0740810
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -0.0727709, upper bound: 0.0740323
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 0, lower bound: -0.0724557, upper bound: 0.0726062
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -0.0724557, upper bound: 0.0742264

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0715002, upper bound: 0.0715002
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0715002, upper bound: 0.0716803
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 15

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0724444, upper bound: 0.0728093
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0724444, upper bound: 0.0728093
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 39

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0717624, upper bound: 0.0723741
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0717624, upper bound: 0.0723741
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0717624, upper bound: 0.0722513
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0717624, upper bound: 0.0719298
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0723034, upper bound: 0.0723882
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0723034, upper bound: 0.0740921
time: 0.31 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.57 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.57
Output dim: 0, lower bound: -0.0715002, upper bound: 0.0715002
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.57
Output dim: 0, lower bound: -0.0715002, upper bound: 0.0716803
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.57
Output dim: 0, lower bound: -0.0724444, upper bound: 0.0728093
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.57
Output dim: 0, lower bound: -0.0724444, upper bound: 0.0728093
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.57
Output dim: 0, lower bound: -0.0717624, upper bound: 0.0723741
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.57
Output dim: 0, lower bound: -0.0717624, upper bound: 0.0723741
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.57
Output dim: 0, lower bound: -0.0717624, upper bound: 0.0722513
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.57
Output dim: 0, lower bound: -0.0717624, upper bound: 0.0719298
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.57
Output dim: 0, lower bound: -0.0723034, upper bound: 0.0723882
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.57
Output dim: 0, lower bound: -0.0723034, upper bound: 0.0740921

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0722366, upper bound: 0.0740259
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0722366, upper bound: 0.0737392
time: 0.33 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 3.09 seconds
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.09
Output dim: 0, lower bound: -0.0722366, upper bound: 0.0740259
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.09
Output dim: 0, lower bound: -0.0722366, upper bound: 0.0737392

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 15

### Candidate
type: RSZ, layer: 3, pos: 26

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 39

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 49

### Candidate
type: RSZ, layer: 3, pos: 7

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721416, upper bound: 0.0721416
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721416, upper bound: 0.0723914
time: 0.32 seconds

## Summary of splitting (split count: 9)
- Time for RS candidates: 4.00 seconds
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 4.00
Output dim: 0, lower bound: -0.0721416, upper bound: 0.0721416
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 4.00
Output dim: 0, lower bound: -0.0721416, upper bound: 0.0723914
Binary search (step 6): status=Status.VERIFIED, low=0.1805223, high=0.1818182, mid=0.1805223, abs_max=0.09181444346904755
rel_dist={0: [-0.07549650593465267, 0.07549650593465268]}

## Binary search (step 7) starts
Candidate diff: 0.1811702


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0751693, upper bound: 0.0751693
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0751693, upper bound: 0.0751743
time: 0.29 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.61 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.61
Output dim: 0, lower bound: -0.0751693, upper bound: 0.0751693
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.61
Output dim: 0, lower bound: -0.0751693, upper bound: 0.0751743

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0750934, upper bound: 0.0750927
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0750927, upper bound: 0.0750934
time: 0.28 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0734733
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0751743
time: 0.32 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.49 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.49
Output dim: 0, lower bound: -0.0750934, upper bound: 0.0750927
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.49
Output dim: 0, lower bound: -0.0750927, upper bound: 0.0750934
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.49
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0734733
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.49
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0751743

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0736330
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0751000, upper bound: 0.0733262
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961
time: 0.30 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.51 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.51
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0736330
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.51
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.51
Output dim: 0, lower bound: -0.0751000, upper bound: 0.0733262
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.51
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.51
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.51
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0730615, upper bound: 0.0730888
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0730615, upper bound: 0.0730888
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0730980, upper bound: 0.0731104
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0748469, upper bound: 0.0731085
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0740920
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749010
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0731438, upper bound: 0.0732505
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731438, upper bound: 0.0749599
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0730615, upper bound: 0.0730884
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0730741, upper bound: 0.0730884
time: 0.32 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.53 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.53
Output dim: 0, lower bound: -0.0730615, upper bound: 0.0730888
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.53
Output dim: 0, lower bound: -0.0730615, upper bound: 0.0730888
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.53
Output dim: 0, lower bound: -0.0730980, upper bound: 0.0731104
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.53
Output dim: 0, lower bound: -0.0748469, upper bound: 0.0731085
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.53
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0740920
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.53
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749010
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.53
Output dim: 0, lower bound: -0.0731438, upper bound: 0.0732505
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.53
Output dim: 0, lower bound: -0.0731438, upper bound: 0.0749599
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.53
Output dim: 0, lower bound: -0.0730615, upper bound: 0.0730884
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.53
Output dim: 0, lower bound: -0.0730741, upper bound: 0.0730884

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0724026, upper bound: 0.0719913
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0719857, upper bound: 0.0719943
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0730860, upper bound: 0.0730860
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0730860, upper bound: 0.0739501
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730559, upper bound: 0.0748079
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0730559, upper bound: 0.0734543
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728619, upper bound: 0.0736910
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728619, upper bound: 0.0746906
time: 0.30 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.51 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 0, lower bound: -0.0724026, upper bound: 0.0719913
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 0, lower bound: -0.0719857, upper bound: 0.0719943
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 0, lower bound: -0.0730860, upper bound: 0.0730860
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 0, lower bound: -0.0730860, upper bound: 0.0739501
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.51
Output dim: 0, lower bound: -0.0730559, upper bound: 0.0748079
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 0, lower bound: -0.0730559, upper bound: 0.0734543
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 0, lower bound: -0.0728619, upper bound: 0.0736910
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.51
Output dim: 0, lower bound: -0.0728619, upper bound: 0.0746906

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 26

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727945, upper bound: 0.0728080
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727945, upper bound: 0.0728080
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0727860, upper bound: 0.0740713
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0727860, upper bound: 0.0745640
time: 0.32 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.56 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.56
Output dim: 0, lower bound: -0.0727945, upper bound: 0.0728080
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.56
Output dim: 0, lower bound: -0.0727945, upper bound: 0.0728080
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.56
Output dim: 0, lower bound: -0.0727860, upper bound: 0.0740713
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.56
Output dim: 0, lower bound: -0.0727860, upper bound: 0.0745640

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0726048, upper bound: 0.0740610
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0726048, upper bound: 0.0739058
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0727232, upper bound: 0.0745000
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727232, upper bound: 0.0728051
time: 0.31 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.98 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.98
Output dim: 0, lower bound: -0.0726048, upper bound: 0.0740610
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.98
Output dim: 0, lower bound: -0.0726048, upper bound: 0.0739058
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.98
Output dim: 0, lower bound: -0.0727232, upper bound: 0.0745000
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.98
Output dim: 0, lower bound: -0.0727232, upper bound: 0.0728051

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0725391, upper bound: 0.0739913
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0725391, upper bound: 0.0727292
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0724874, upper bound: 0.0724874
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0724874, upper bound: 0.0725111
time: 0.31 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 3.43 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.43
Output dim: 0, lower bound: -0.0725391, upper bound: 0.0739913
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.43
Output dim: 0, lower bound: -0.0725391, upper bound: 0.0727292
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.43
Output dim: 0, lower bound: -0.0724874, upper bound: 0.0724874
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.43
Output dim: 0, lower bound: -0.0724874, upper bound: 0.0725111

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0722366, upper bound: 0.0737248
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0722366, upper bound: 0.0722487
time: 0.31 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 3.52 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.52
Output dim: 0, lower bound: -0.0722366, upper bound: 0.0737248
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.52
Output dim: 0, lower bound: -0.0722366, upper bound: 0.0722487
Binary search (step 7): status=Status.VERIFIED, low=0.1811702, high=0.1818182, mid=0.1811702, abs_max=0.09181444346904755
rel_dist={0: [-0.07549650593465267, 0.07549650593465268]}

## Binary search (step 8) starts
Candidate diff: 0.1814942


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0741131, upper bound: 0.0741131
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0741131, upper bound: 0.0754287
time: 0.30 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.61 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.61
Output dim: 0, lower bound: -0.0741131, upper bound: 0.0741131
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.61
Output dim: 0, lower bound: -0.0741131, upper bound: 0.0754287

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0751743, upper bound: 0.0737033
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0734733
time: 0.32 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233
time: 0.30 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.52 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.52
Output dim: 0, lower bound: -0.0751743, upper bound: 0.0737033
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.52
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0734733
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.52
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.52
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0736330
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0733262
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
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

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

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
- Time for RS candidates: 2.54 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.54
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0736330
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.54
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0733262
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.54
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.54
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.54
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.54
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749547
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749518
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749836
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
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

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0732285, upper bound: 0.0750105
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0732285, upper bound: 0.0737973
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731010, upper bound: 0.0748469
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0731448, upper bound: 0.0730980
time: 0.33 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.56 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.56
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749547
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.56
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749518
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.56
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749836
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.56
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749810
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.56
Output dim: 0, lower bound: -0.0732285, upper bound: 0.0750105
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.56
Output dim: 0, lower bound: -0.0732285, upper bound: 0.0737973
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.56
Output dim: 0, lower bound: -0.0731010, upper bound: 0.0748469
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.56
Output dim: 0, lower bound: -0.0731448, upper bound: 0.0730980

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0729207, upper bound: 0.0747715
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0729207, upper bound: 0.0729207
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730559, upper bound: 0.0748730
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0730559, upper bound: 0.0730559
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0729207, upper bound: 0.0747812
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0729229, upper bound: 0.0729207
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0728390
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0746147
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730160, upper bound: 0.0747572
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0730160, upper bound: 0.0730238
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727890, upper bound: 0.0727890
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727890, upper bound: 0.0729773
time: 0.29 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.54 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.54
Output dim: 0, lower bound: -0.0729207, upper bound: 0.0747715
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.54
Output dim: 0, lower bound: -0.0729207, upper bound: 0.0729207
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.54
Output dim: 0, lower bound: -0.0730559, upper bound: 0.0748730
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.54
Output dim: 0, lower bound: -0.0730559, upper bound: 0.0730559
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.54
Output dim: 0, lower bound: -0.0729207, upper bound: 0.0747812
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.54
Output dim: 0, lower bound: -0.0729229, upper bound: 0.0729207
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.54
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0728390
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.54
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0746147
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.54
Output dim: 0, lower bound: -0.0730160, upper bound: 0.0747572
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.54
Output dim: 0, lower bound: -0.0730160, upper bound: 0.0730238
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.54
Output dim: 0, lower bound: -0.0727890, upper bound: 0.0727890
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.54
Output dim: 0, lower bound: -0.0727890, upper bound: 0.0729773

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0725550, upper bound: 0.0741783
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0725550, upper bound: 0.0743084
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0727665, upper bound: 0.0742567
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0727665, upper bound: 0.0744681
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0726491, upper bound: 0.0730588
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0726491, upper bound: 0.0730588
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0726935, upper bound: 0.0727474
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0726935, upper bound: 0.0745169
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0729546, upper bound: 0.0734904
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0729546, upper bound: 0.0746222
time: 0.31 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.55 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -0.0725550, upper bound: 0.0741783
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -0.0725550, upper bound: 0.0743084
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -0.0727665, upper bound: 0.0742567
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -0.0727665, upper bound: 0.0744681
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 0, lower bound: -0.0726491, upper bound: 0.0730588
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 0, lower bound: -0.0726491, upper bound: 0.0730588
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 0, lower bound: -0.0726935, upper bound: 0.0727474
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -0.0726935, upper bound: 0.0745169
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 0, lower bound: -0.0729546, upper bound: 0.0734904
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -0.0729546, upper bound: 0.0746222

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0724021, upper bound: 0.0728014
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0724021, upper bound: 0.0740314
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0724522, upper bound: 0.0727834
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0724522, upper bound: 0.0727834
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0724963, upper bound: 0.0724711
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0724954, upper bound: 0.0724833
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0724711, upper bound: 0.0724711
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0724711, upper bound: 0.0724842
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0726313, upper bound: 0.0744517
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0726313, upper bound: 0.0726313
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0725693, upper bound: 0.0736160
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0725687, upper bound: 0.0740201
time: 0.37 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.74 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.74
Output dim: 0, lower bound: -0.0724021, upper bound: 0.0728014
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -0.0724021, upper bound: 0.0740314
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.74
Output dim: 0, lower bound: -0.0724522, upper bound: 0.0727834
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.74
Output dim: 0, lower bound: -0.0724522, upper bound: 0.0727834
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.74
Output dim: 0, lower bound: -0.0724963, upper bound: 0.0724711
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.74
Output dim: 0, lower bound: -0.0724954, upper bound: 0.0724833
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.74
Output dim: 0, lower bound: -0.0724711, upper bound: 0.0724711
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.74
Output dim: 0, lower bound: -0.0724711, upper bound: 0.0724842
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -0.0726313, upper bound: 0.0744517
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.74
Output dim: 0, lower bound: -0.0726313, upper bound: 0.0726313
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.74
Output dim: 0, lower bound: -0.0725693, upper bound: 0.0736160
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -0.0725687, upper bound: 0.0740201

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Candidate
type: RSZ, layer: 3, pos: 15

### Candidate
type: RSZ, layer: 3, pos: 26

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0723028, upper bound: 0.0726075
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0723028, upper bound: 0.0726075
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0723625, upper bound: 0.0723625
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0723625, upper bound: 0.0723895
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0723792, upper bound: 0.0731016
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0723792, upper bound: 0.0737993
time: 0.32 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.60 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.60
Output dim: 0, lower bound: -0.0723028, upper bound: 0.0726075
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.60
Output dim: 0, lower bound: -0.0723028, upper bound: 0.0726075
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.60
Output dim: 0, lower bound: -0.0723625, upper bound: 0.0723625
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.60
Output dim: 0, lower bound: -0.0723625, upper bound: 0.0723895
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.60
Output dim: 0, lower bound: -0.0723792, upper bound: 0.0731016
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.60
Output dim: 0, lower bound: -0.0723792, upper bound: 0.0737993
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

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753536
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753784
time: 0.37 seconds

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

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0750934, upper bound: 0.0750927
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0750934, upper bound: 0.0751000
time: 0.34 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753228, upper bound: 0.0734268
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233
time: 0.35 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.65 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.65
Output dim: 0, lower bound: -0.0750934, upper bound: 0.0750927
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.65
Output dim: 0, lower bound: -0.0750934, upper bound: 0.0751000
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.65
Output dim: 0, lower bound: -0.0753228, upper bound: 0.0734268
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.65
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0750961, upper bound: 0.0736330
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0734262
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734262, upper bound: 0.0733262
time: 0.30 seconds

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

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

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
- Time for RS candidates: 2.46 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 0, lower bound: -0.0750961, upper bound: 0.0736330
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.46
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0734262
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.46
Output dim: 0, lower bound: -0.0734262, upper bound: 0.0733262
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.46
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0733124
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734931, upper bound: 0.0731184
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0731094, upper bound: 0.0731094
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730145, upper bound: 0.0745164
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730145, upper bound: 0.0747252
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
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730980, upper bound: 0.0749041
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0731104, upper bound: 0.0730980
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0731094, upper bound: 0.0735058
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0731114, upper bound: 0.0735532
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
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0731094, upper bound: 0.0731094
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0731184, upper bound: 0.0734931
time: 0.30 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.50 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.50
Output dim: 0, lower bound: -0.0734931, upper bound: 0.0731184
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.50
Output dim: 0, lower bound: -0.0731094, upper bound: 0.0731094
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.50
Output dim: 0, lower bound: -0.0730145, upper bound: 0.0745164
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.50
Output dim: 0, lower bound: -0.0730145, upper bound: 0.0747252
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.50
Output dim: 0, lower bound: -0.0730980, upper bound: 0.0749041
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.50
Output dim: 0, lower bound: -0.0731104, upper bound: 0.0730980
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.50
Output dim: 0, lower bound: -0.0731094, upper bound: 0.0735058
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.50
Output dim: 0, lower bound: -0.0731114, upper bound: 0.0735532
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.50
Output dim: 0, lower bound: -0.0731094, upper bound: 0.0731094
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.50
Output dim: 0, lower bound: -0.0731184, upper bound: 0.0734931

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0729442, upper bound: 0.0731196
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0729442, upper bound: 0.0740442
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0727463, upper bound: 0.0744861
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727463, upper bound: 0.0727463
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730160, upper bound: 0.0748264
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0730248, upper bound: 0.0731375
time: 0.30 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.51 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 0, lower bound: -0.0729442, upper bound: 0.0731196
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.51
Output dim: 0, lower bound: -0.0729442, upper bound: 0.0740442
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.51
Output dim: 0, lower bound: -0.0727463, upper bound: 0.0744861
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 0, lower bound: -0.0727463, upper bound: 0.0727463
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.51
Output dim: 0, lower bound: -0.0730160, upper bound: 0.0748264
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 0, lower bound: -0.0730248, upper bound: 0.0731375

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0726471, upper bound: 0.0739297
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0726471, upper bound: 0.0726471
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0726681, upper bound: 0.0744076
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0726681, upper bound: 0.0726681
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728400, upper bound: 0.0730307
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728400, upper bound: 0.0746538
time: 0.31 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.54 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.54
Output dim: 0, lower bound: -0.0726471, upper bound: 0.0739297
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.54
Output dim: 0, lower bound: -0.0726471, upper bound: 0.0726471
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -0.0726681, upper bound: 0.0744076
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.54
Output dim: 0, lower bound: -0.0726681, upper bound: 0.0726681
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.54
Output dim: 0, lower bound: -0.0728400, upper bound: 0.0730307
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.54
Output dim: 0, lower bound: -0.0728400, upper bound: 0.0746538

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0725595, upper bound: 0.0727559
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0725595, upper bound: 0.0727559
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0725247, upper bound: 0.0734791
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0725247, upper bound: 0.0743033
time: 0.30 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.54 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.54
Output dim: 0, lower bound: -0.0725595, upper bound: 0.0727559
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.54
Output dim: 0, lower bound: -0.0725595, upper bound: 0.0727559
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.54
Output dim: 0, lower bound: -0.0725247, upper bound: 0.0734791
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 0, lower bound: -0.0725247, upper bound: 0.0743033

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0723359, upper bound: 0.0741810
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0723359, upper bound: 0.0741293
time: 0.31 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.53 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 0, lower bound: -0.0723359, upper bound: 0.0741810
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 0, lower bound: -0.0723359, upper bound: 0.0741293

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0722444, upper bound: 0.0726567
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0722444, upper bound: 0.0726590
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 48

### Candidate
type: RSZ, layer: 3, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0722366, upper bound: 0.0736881
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0722366, upper bound: 0.0739907
time: 0.30 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 3.32 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.32
Output dim: 0, lower bound: -0.0722444, upper bound: 0.0726567
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.32
Output dim: 0, lower bound: -0.0722444, upper bound: 0.0726590
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.32
Output dim: 0, lower bound: -0.0722366, upper bound: 0.0736881
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.32
Output dim: 0, lower bound: -0.0722366, upper bound: 0.0739907

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 15

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721416, upper bound: 0.0723534
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0721416, upper bound: 0.0724055
time: 0.30 seconds

## Summary of splitting (split count: 9)
- Time for RS candidates: 3.80 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 3.80
Output dim: 0, lower bound: -0.0721416, upper bound: 0.0723534
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 3.80
Output dim: 0, lower bound: -0.0721416, upper bound: 0.0724055
Binary search (step 9): status=Status.VERIFIED, low=0.1816562, high=0.1818182, mid=0.1816562, abs_max=0.09181444346904755
rel_dist={0: [-0.07549650593465267, 0.07549650593465268]}

## Binary search (step 10) starts
Candidate diff: 0.1817372


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0741131, upper bound: 0.0741131
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0741131, upper bound: 0.0754287
time: 0.29 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.60 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.60
Output dim: 0, lower bound: -0.0741131, upper bound: 0.0741131
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.60
Output dim: 0, lower bound: -0.0741131, upper bound: 0.0754287

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0739373
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0734268
time: 0.31 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233
time: 0.29 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.37 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.37
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0739373
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.37
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0734268
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.37
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.37
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
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

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

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
- Time for RS candidates: 2.38 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.38
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.38
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.38
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.38
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0722400, upper bound: 0.0722848
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0722400, upper bound: 0.0724194
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0722488, upper bound: 0.0725979
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0722488, upper bound: 0.0725979
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731010, upper bound: 0.0748421
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0731448, upper bound: 0.0731075
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0732285, upper bound: 0.0750137
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0732285, upper bound: 0.0744276
time: 0.29 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.39 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.39
Output dim: 0, lower bound: -0.0722400, upper bound: 0.0722848
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.39
Output dim: 0, lower bound: -0.0722400, upper bound: 0.0724194
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.39
Output dim: 0, lower bound: -0.0722488, upper bound: 0.0725979
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.39
Output dim: 0, lower bound: -0.0722488, upper bound: 0.0725979
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -0.0731010, upper bound: 0.0748421
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.39
Output dim: 0, lower bound: -0.0731448, upper bound: 0.0731075
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -0.0732285, upper bound: 0.0750137
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -0.0732285, upper bound: 0.0744276

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730160, upper bound: 0.0747572
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0730243, upper bound: 0.0737053
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
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730559, upper bound: 0.0748939
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730559, upper bound: 0.0748912
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0730698, upper bound: 0.0731302
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730698, upper bound: 0.0743009
time: 0.30 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.41 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.41
Output dim: 0, lower bound: -0.0730160, upper bound: 0.0747572
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.41
Output dim: 0, lower bound: -0.0730243, upper bound: 0.0737053
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.41
Output dim: 0, lower bound: -0.0730559, upper bound: 0.0748939
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.41
Output dim: 0, lower bound: -0.0730559, upper bound: 0.0748912
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.41
Output dim: 0, lower bound: -0.0730698, upper bound: 0.0731302
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.41
Output dim: 0, lower bound: -0.0730698, upper bound: 0.0743009

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0719119, upper bound: 0.0721287
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0719119, upper bound: 0.0722351
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728898, upper bound: 0.0728898
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728898, upper bound: 0.0732631
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0729057, upper bound: 0.0729528
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0729057, upper bound: 0.0747524
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728596, upper bound: 0.0728851
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728596, upper bound: 0.0728867
time: 0.31 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.44 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.44
Output dim: 0, lower bound: -0.0719119, upper bound: 0.0721287
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.44
Output dim: 0, lower bound: -0.0719119, upper bound: 0.0722351
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.44
Output dim: 0, lower bound: -0.0728898, upper bound: 0.0728898
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.44
Output dim: 0, lower bound: -0.0728898, upper bound: 0.0732631
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.44
Output dim: 0, lower bound: -0.0729057, upper bound: 0.0729528
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.44
Output dim: 0, lower bound: -0.0729057, upper bound: 0.0747524
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.44
Output dim: 0, lower bound: -0.0728596, upper bound: 0.0728851
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.44
Output dim: 0, lower bound: -0.0728596, upper bound: 0.0728867

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728480, upper bound: 0.0741368
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728480, upper bound: 0.0746972
time: 0.29 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.47 seconds
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.47
Output dim: 0, lower bound: -0.0728480, upper bound: 0.0741368
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.47
Output dim: 0, lower bound: -0.0728480, upper bound: 0.0746972

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716713, upper bound: 0.0721899
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716713, upper bound: 0.0721899
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0726489, upper bound: 0.0726489
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0726489, upper bound: 0.0731459
time: 0.31 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.51 seconds
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -0.0716713, upper bound: 0.0721899
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -0.0716713, upper bound: 0.0721899
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -0.0726489, upper bound: 0.0726489
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -0.0726489, upper bound: 0.0731459
Binary search (step 10): status=Status.VERIFIED, low=0.1817372, high=0.1818182, mid=0.1817372, abs_max=0.09181444346904755
rel_dist={0: [-0.07549650593465267, 0.07549650593465268]}

## Binary search (step 11) starts
Candidate diff: 0.1817777


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0751693, upper bound: 0.0751693
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0751693, upper bound: 0.0751743
time: 0.28 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.60 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.60
Output dim: 0, lower bound: -0.0751693, upper bound: 0.0751693
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.60
Output dim: 0, lower bound: -0.0751693, upper bound: 0.0751743

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
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0737033
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0751693
time: 0.30 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0750934, upper bound: 0.0751000
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0750927, upper bound: 0.0750980
time: 0.29 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.43 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.43
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0737033
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.43
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0751693
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.43
Output dim: 0, lower bound: -0.0750934, upper bound: 0.0751000
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.43
Output dim: 0, lower bound: -0.0750927, upper bound: 0.0750980

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0733262, upper bound: 0.0734262
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
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

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0733124
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961
time: 0.28 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.48 seconds
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.48
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.48
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.48
Output dim: 0, lower bound: -0.0733262, upper bound: 0.0734262
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.48
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.48
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0733124
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.48
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
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731438, upper bound: 0.0744930
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731438, upper bound: 0.0749422
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730980, upper bound: 0.0748421
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0731448, upper bound: 0.0731075
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0732624, upper bound: 0.0744218
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0732624, upper bound: 0.0749948
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
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0731438, upper bound: 0.0732379
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731648, upper bound: 0.0749539
time: 0.29 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.44 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0731438, upper bound: 0.0744930
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0731438, upper bound: 0.0749422
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0730980, upper bound: 0.0748421
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0731448, upper bound: 0.0731075
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0732624, upper bound: 0.0744218
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0732624, upper bound: 0.0749948
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0731438, upper bound: 0.0732379
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0731648, upper bound: 0.0749539

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728619, upper bound: 0.0732984
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728619, upper bound: 0.0732268
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0729281, upper bound: 0.0729281
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0729281, upper bound: 0.0729369
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
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0729209, upper bound: 0.0737147
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0729209, upper bound: 0.0746722
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0729442, upper bound: 0.0729979
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0729442, upper bound: 0.0741860
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0730901, upper bound: 0.0730901
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730901, upper bound: 0.0748550
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730901, upper bound: 0.0743136
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730901, upper bound: 0.0748546
time: 0.35 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.66 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0728619, upper bound: 0.0732984
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0728619, upper bound: 0.0732268
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0729281, upper bound: 0.0729281
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0729281, upper bound: 0.0729369
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0729209, upper bound: 0.0737147
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0729209, upper bound: 0.0746722
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0729442, upper bound: 0.0729979
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0729442, upper bound: 0.0741860
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0730901, upper bound: 0.0730901
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0730901, upper bound: 0.0748550
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0730901, upper bound: 0.0743136
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0730901, upper bound: 0.0748546

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728521, upper bound: 0.0734161
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728521, upper bound: 0.0745099
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0726744, upper bound: 0.0726744
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0726744, upper bound: 0.0726744
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0719027, upper bound: 0.0722172
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0719027, upper bound: 0.0722453
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0729256, upper bound: 0.0742362
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0729256, upper bound: 0.0742147
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0719029, upper bound: 0.0725361
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0719029, upper bound: 0.0725361
time: 0.33 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.57 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0728521, upper bound: 0.0734161
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0728521, upper bound: 0.0745099
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0726744, upper bound: 0.0726744
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0726744, upper bound: 0.0726744
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0719027, upper bound: 0.0722172
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0719027, upper bound: 0.0722453
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0729256, upper bound: 0.0742362
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0729256, upper bound: 0.0742147
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0719029, upper bound: 0.0725361
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0719029, upper bound: 0.0725361

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0724902, upper bound: 0.0735416
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0724902, upper bound: 0.0739441
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0726849, upper bound: 0.0740065
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728993, upper bound: 0.0726899
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0717195, upper bound: 0.0722384
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0717195, upper bound: 0.0722384
time: 0.33 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.61 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 0, lower bound: -0.0724902, upper bound: 0.0735416
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 0, lower bound: -0.0724902, upper bound: 0.0739441
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 0, lower bound: -0.0726849, upper bound: 0.0740065
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 0, lower bound: -0.0728993, upper bound: 0.0726899
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 0, lower bound: -0.0717195, upper bound: 0.0722384
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 0, lower bound: -0.0717195, upper bound: 0.0722384

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0726028, upper bound: 0.0739400
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0726043, upper bound: 0.0738872
time: 0.35 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.61 seconds
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -0.0726028, upper bound: 0.0739400
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -0.0726043, upper bound: 0.0738872
Binary search (step 11): status=Status.VERIFIED, low=0.1817777, high=0.1818182, mid=0.1817777, abs_max=0.09181444346904755
rel_dist={0: [-0.07549650593465267, 0.07549650593465268]}

## Binary search (step 12) starts
Candidate diff: 0.1817979


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0754287, upper bound: 0.0741131
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0741131, upper bound: 0.0754287
time: 0.29 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.61 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.61
Output dim: 0, lower bound: -0.0754287, upper bound: 0.0741131
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.61
Output dim: 0, lower bound: -0.0741131, upper bound: 0.0754287

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0737033
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0734733
time: 0.32 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0751693
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0751743
time: 0.30 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.37 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.37
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0737033
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.37
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0734733
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.37
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0751693
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.37
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0751743

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961
time: 0.27 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.37 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.37
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.37
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.37
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.37
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0732624, upper bound: 0.0737588
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0732624, upper bound: 0.0749328
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0722400, upper bound: 0.0723976
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0722400, upper bound: 0.0724417
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0722488, upper bound: 0.0725979
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0722488, upper bound: 0.0725979
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
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0732624, upper bound: 0.0744456
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0732624, upper bound: 0.0749951
time: 0.29 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.41 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.41
Output dim: 0, lower bound: -0.0732624, upper bound: 0.0737588
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.41
Output dim: 0, lower bound: -0.0732624, upper bound: 0.0749328
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.41
Output dim: 0, lower bound: -0.0722400, upper bound: 0.0723976
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.41
Output dim: 0, lower bound: -0.0722400, upper bound: 0.0724417
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.41
Output dim: 0, lower bound: -0.0722488, upper bound: 0.0725979
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.41
Output dim: 0, lower bound: -0.0722488, upper bound: 0.0725979
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.41
Output dim: 0, lower bound: -0.0732624, upper bound: 0.0744456
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.41
Output dim: 0, lower bound: -0.0732624, upper bound: 0.0749951

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730860, upper bound: 0.0748233
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730860, upper bound: 0.0748160
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0730901, upper bound: 0.0732138
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730901, upper bound: 0.0743136
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0730039, upper bound: 0.0730039
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0730039, upper bound: 0.0733604
time: 0.31 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.65 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.65
Output dim: 0, lower bound: -0.0730860, upper bound: 0.0748233
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.65
Output dim: 0, lower bound: -0.0730860, upper bound: 0.0748160
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.65
Output dim: 0, lower bound: -0.0730901, upper bound: 0.0732138
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.65
Output dim: 0, lower bound: -0.0730901, upper bound: 0.0743136
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.65
Output dim: 0, lower bound: -0.0730039, upper bound: 0.0730039
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.65
Output dim: 0, lower bound: -0.0730039, upper bound: 0.0733604

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728531, upper bound: 0.0734720
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728531, upper bound: 0.0734722
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 39

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0729256, upper bound: 0.0740939
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0729256, upper bound: 0.0746656
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728426, upper bound: 0.0728924
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728426, upper bound: 0.0728934
time: 0.31 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.67 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.67
Output dim: 0, lower bound: -0.0728531, upper bound: 0.0734720
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.67
Output dim: 0, lower bound: -0.0728531, upper bound: 0.0734722
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -0.0729256, upper bound: 0.0740939
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -0.0729256, upper bound: 0.0746656
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.67
Output dim: 0, lower bound: -0.0728426, upper bound: 0.0728924
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.67
Output dim: 0, lower bound: -0.0728426, upper bound: 0.0728934

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 39

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0717195, upper bound: 0.0717195
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0717195, upper bound: 0.0717353
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0726048, upper bound: 0.0737993
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0726048, upper bound: 0.0741282
time: 0.31 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.47 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.47
Output dim: 0, lower bound: -0.0717195, upper bound: 0.0717195
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.47
Output dim: 0, lower bound: -0.0717195, upper bound: 0.0717353
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.47
Output dim: 0, lower bound: -0.0726048, upper bound: 0.0737993
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.47
Output dim: 0, lower bound: -0.0726048, upper bound: 0.0741282

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0725391, upper bound: 0.0740542
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0725391, upper bound: 0.0725391
time: 0.33 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.53 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 0, lower bound: -0.0725391, upper bound: 0.0740542
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.53
Output dim: 0, lower bound: -0.0725391, upper bound: 0.0725391

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 39

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0722802, upper bound: 0.0722802
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0722802, upper bound: 0.0723022
time: 0.30 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 3.37 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.37
Output dim: 0, lower bound: -0.0722802, upper bound: 0.0722802
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.37
Output dim: 0, lower bound: -0.0722802, upper bound: 0.0723022
Binary search (step 12): status=Status.VERIFIED, low=0.1817979, high=0.1818182, mid=0.1817979, abs_max=0.09181444346904755
rel_dist={0: [-0.07549650593465267, 0.07549650593465268]}

## Binary search (step 13) starts
Candidate diff: 0.1818081


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0751693, upper bound: 0.0751693
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0751693, upper bound: 0.0751743
time: 0.28 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.59 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.59
Output dim: 0, lower bound: -0.0751693, upper bound: 0.0751693
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.59
Output dim: 0, lower bound: -0.0751693, upper bound: 0.0751743

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
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0737033
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0751693
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
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0750934, upper bound: 0.0751000
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0750927, upper bound: 0.0750980
time: 0.30 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.40 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.40
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0737033
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.40
Output dim: 0, lower bound: -0.0734733, upper bound: 0.0751693
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.40
Output dim: 0, lower bound: -0.0750934, upper bound: 0.0751000
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.40
Output dim: 0, lower bound: -0.0750927, upper bound: 0.0750980

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0734262
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0750927, upper bound: 0.0733124
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961
time: 0.29 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.52 seconds
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0734262
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0750927, upper bound: 0.0733124
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0732285, upper bound: 0.0750152
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0732285, upper bound: 0.0732395
time: 0.31 seconds

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
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0740920
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749010
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0731094, upper bound: 0.0735845
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0731125, upper bound: 0.0735916
time: 0.32 seconds

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
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0749518, upper bound: 0.0731370
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0749547, upper bound: 0.0731368
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
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0730615, upper bound: 0.0730884
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0730615, upper bound: 0.0730884
time: 0.30 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.42 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0732285, upper bound: 0.0750152
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0732285, upper bound: 0.0732395
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0740920
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749010
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0731094, upper bound: 0.0735845
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0731125, upper bound: 0.0735916
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0749518, upper bound: 0.0731370
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0749547, upper bound: 0.0731368
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0730615, upper bound: 0.0730884
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0730615, upper bound: 0.0730884

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730698, upper bound: 0.0744149
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730698, upper bound: 0.0748615
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0729555, upper bound: 0.0729555
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0729555, upper bound: 0.0731967
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0729835, upper bound: 0.0736470
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0729835, upper bound: 0.0747569
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
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728695, upper bound: 0.0728695
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728695, upper bound: 0.0728695
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
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0748233, upper bound: 0.0730860
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0730860, upper bound: 0.0730860
time: 0.30 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.46 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.46
Output dim: 0, lower bound: -0.0730698, upper bound: 0.0744149
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.46
Output dim: 0, lower bound: -0.0730698, upper bound: 0.0748615
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.46
Output dim: 0, lower bound: -0.0729555, upper bound: 0.0729555
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.46
Output dim: 0, lower bound: -0.0729555, upper bound: 0.0731967
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.46
Output dim: 0, lower bound: -0.0729835, upper bound: 0.0736470
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.46
Output dim: 0, lower bound: -0.0729835, upper bound: 0.0747569
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.46
Output dim: 0, lower bound: -0.0728695, upper bound: 0.0728695
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.46
Output dim: 0, lower bound: -0.0728695, upper bound: 0.0728695
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.46
Output dim: 0, lower bound: -0.0748233, upper bound: 0.0730860
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.46
Output dim: 0, lower bound: -0.0730860, upper bound: 0.0730860

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0729057, upper bound: 0.0741890
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0729057, upper bound: 0.0741960
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728596, upper bound: 0.0728596
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728596, upper bound: 0.0728799
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727452, upper bound: 0.0727452
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727452, upper bound: 0.0727452
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0746587, upper bound: 0.0729256
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0729256, upper bound: 0.0729256
time: 0.30 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.43 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.43
Output dim: 0, lower bound: -0.0729057, upper bound: 0.0741890
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.43
Output dim: 0, lower bound: -0.0729057, upper bound: 0.0741960
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.43
Output dim: 0, lower bound: -0.0728596, upper bound: 0.0728596
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.43
Output dim: 0, lower bound: -0.0728596, upper bound: 0.0728799
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.43
Output dim: 0, lower bound: -0.0727452, upper bound: 0.0727452
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.43
Output dim: 0, lower bound: -0.0727452, upper bound: 0.0727452
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.43
Output dim: 0, lower bound: -0.0746587, upper bound: 0.0729256
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.43
Output dim: 0, lower bound: -0.0729256, upper bound: 0.0729256

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728480, upper bound: 0.0731577
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728480, upper bound: 0.0739941
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 48

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0726636, upper bound: 0.0726636
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0726636, upper bound: 0.0726638
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0726048, upper bound: 0.0726048
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0726048, upper bound: 0.0726048
time: 0.30 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.48 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0728480, upper bound: 0.0731577
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0728480, upper bound: 0.0739941
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0726636, upper bound: 0.0726636
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0726636, upper bound: 0.0726638
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0726048, upper bound: 0.0726048
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0726048, upper bound: 0.0726048

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0725828, upper bound: 0.0725828
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0725828, upper bound: 0.0725828
time: 0.30 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.47 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.47
Output dim: 0, lower bound: -0.0725828, upper bound: 0.0725828
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.47
Output dim: 0, lower bound: -0.0725828, upper bound: 0.0725828
Binary search (step 13): status=Status.VERIFIED, low=0.1818081, high=0.1818182, mid=0.1818081, abs_max=0.09181444346904755
rel_dist={0: [-0.07549650593465267, 0.07549650593465268]}

## Binary search (step 14) starts
Candidate diff: 0.1818131


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

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
- Time for RS candidates: 0.60 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.60
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753536
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.60
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
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0750934, upper bound: 0.0750927
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0750927, upper bound: 0.0751000
time: 0.28 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0750927, upper bound: 0.0750934
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0750927, upper bound: 0.0750980
time: 0.30 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.37 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.37
Output dim: 0, lower bound: -0.0750934, upper bound: 0.0750927
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.37
Output dim: 0, lower bound: -0.0750927, upper bound: 0.0751000
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.37
Output dim: 0, lower bound: -0.0750927, upper bound: 0.0750934
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.37
Output dim: 0, lower bound: -0.0750927, upper bound: 0.0750980

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0736330
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0734262
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
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734262, upper bound: 0.0733262
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
time: 0.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0733124
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961
time: 0.27 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.36 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.36
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0736330
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.36
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.36
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0734262
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.36
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.36
Output dim: 0, lower bound: -0.0734262, upper bound: 0.0733262
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.36
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.36
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0733124
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.36
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730145, upper bound: 0.0745164
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730145, upper bound: 0.0747252
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0722488, upper bound: 0.0725979
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0722488, upper bound: 0.0725979
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731010, upper bound: 0.0748421
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0731448, upper bound: 0.0731075
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0722488, upper bound: 0.0728976
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0722400, upper bound: 0.0728976
time: 0.32 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.43 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.43
Output dim: 0, lower bound: -0.0730145, upper bound: 0.0745164
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.43
Output dim: 0, lower bound: -0.0730145, upper bound: 0.0747252
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.43
Output dim: 0, lower bound: -0.0722488, upper bound: 0.0725979
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.43
Output dim: 0, lower bound: -0.0722488, upper bound: 0.0725979
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.43
Output dim: 0, lower bound: -0.0731010, upper bound: 0.0748421
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.43
Output dim: 0, lower bound: -0.0731448, upper bound: 0.0731075
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.43
Output dim: 0, lower bound: -0.0722488, upper bound: 0.0728976
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.43
Output dim: 0, lower bound: -0.0722400, upper bound: 0.0728976

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727703, upper bound: 0.0727703
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727703, upper bound: 0.0727736
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0745305
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0745404
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0730311, upper bound: 0.0735708
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730311, upper bound: 0.0747175
time: 0.32 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.54 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.54
Output dim: 0, lower bound: -0.0727703, upper bound: 0.0727703
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.54
Output dim: 0, lower bound: -0.0727703, upper bound: 0.0727736
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.54
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0745305
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.54
Output dim: 0, lower bound: -0.0728390, upper bound: 0.0745404
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.54
Output dim: 0, lower bound: -0.0730311, upper bound: 0.0735708
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.54
Output dim: 0, lower bound: -0.0730311, upper bound: 0.0747175

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0726935, upper bound: 0.0728479
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0726935, upper bound: 0.0744150
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0725485, upper bound: 0.0725485
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0725485, upper bound: 0.0725557
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0728521, upper bound: 0.0736718
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728521, upper bound: 0.0745099
time: 0.32 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.13 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0726935, upper bound: 0.0728479
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0726935, upper bound: 0.0744150
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0725485, upper bound: 0.0725485
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0725485, upper bound: 0.0725557
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0728521, upper bound: 0.0736718
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0728521, upper bound: 0.0745099

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0726313, upper bound: 0.0743461
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0726313, upper bound: 0.0726313
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0725147, upper bound: 0.0727200
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0725147, upper bound: 0.0727280
time: 0.32 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.11 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.0726313, upper bound: 0.0743461
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.0726313, upper bound: 0.0726313
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.0725147, upper bound: 0.0727200
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.11
Output dim: 0, lower bound: -0.0725147, upper bound: 0.0727280

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0725507, upper bound: 0.0731130
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0725507, upper bound: 0.0731406
time: 0.31 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.67 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.67
Output dim: 0, lower bound: -0.0725507, upper bound: 0.0731130
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.67
Output dim: 0, lower bound: -0.0725507, upper bound: 0.0731406
Binary search (step 14): status=Status.VERIFIED, low=0.1818131, high=0.1818182, mid=0.1818131, abs_max=0.09181444346904755
rel_dist={0: [-0.07549650593465267, 0.07549650593465268]}

## Binary search (step 15) starts
Candidate diff: 0.1818157


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

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
- Time for RS candidates: 0.63 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.63
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753536
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.63
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753784

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0739373
time: 0.32 seconds

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

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

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
time: 0.31 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.61 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.61
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0739373
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.61
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.61
Output dim: 0, lower bound: -0.0753228, upper bound: 0.0734268
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.61
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0752233

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

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

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0751000, upper bound: 0.0733262
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0733124
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

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
- Time for RS candidates: 2.52 seconds
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0751000, upper bound: 0.0733262
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0733124
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

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0722400, upper bound: 0.0722848
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0722400, upper bound: 0.0724194
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0731094, upper bound: 0.0735845
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0731125, upper bound: 0.0735916
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0731368
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0749836, upper bound: 0.0731368
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0730615, upper bound: 0.0731049
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0730615, upper bound: 0.0731049
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0731094, upper bound: 0.0731094
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0731184, upper bound: 0.0734931
time: 0.30 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.59 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -0.0722400, upper bound: 0.0722848
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -0.0722400, upper bound: 0.0724194
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -0.0731094, upper bound: 0.0735845
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -0.0731125, upper bound: 0.0735916
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0731368
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.59
Output dim: 0, lower bound: -0.0749836, upper bound: 0.0731368
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -0.0730615, upper bound: 0.0731049
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -0.0730615, upper bound: 0.0731049
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -0.0731094, upper bound: 0.0731094
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -0.0731184, upper bound: 0.0734931

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0724923, upper bound: 0.0721101
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0724923, upper bound: 0.0721101
time: 0.30 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.42 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.42
Output dim: 0, lower bound: -0.0724923, upper bound: 0.0721101
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.42
Output dim: 0, lower bound: -0.0724923, upper bound: 0.0721101
Binary search (step 15): status=Status.VERIFIED, low=0.1818157, high=0.1818182, mid=0.1818157, abs_max=0.09181444346904755
rel_dist={0: [-0.07549650593465267, 0.07549650593465268]}

## Binary search (step 16) starts
Candidate diff: 0.1818169


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

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
- Time for RS candidates: 0.59 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.59
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753536
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.59
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

Time for candidate selection: 0.00 seconds

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

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0750927, upper bound: 0.0750934
time: 0.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0750927, upper bound: 0.0750980
time: 0.30 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.36 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.36
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0739373
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.36
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.36
Output dim: 0, lower bound: -0.0750927, upper bound: 0.0750934
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.36
Output dim: 0, lower bound: -0.0750927, upper bound: 0.0750980

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

Time for candidate selection: 0.00 seconds

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

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0751000, upper bound: 0.0733262
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
time: 0.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0733124
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961
time: 0.29 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.40 seconds
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.40
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.40
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.40
Output dim: 0, lower bound: -0.0751000, upper bound: 0.0733262
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.40
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.40
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0733124
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.40
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0730615, upper bound: 0.0730888
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0730615, upper bound: 0.0730888
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0730615, upper bound: 0.0730615
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0730615, upper bound: 0.0730615
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0735916, upper bound: 0.0731125
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0735845, upper bound: 0.0731103
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0731094, upper bound: 0.0735058
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0731114, upper bound: 0.0735532
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
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0722489, upper bound: 0.0728976
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0722400, upper bound: 0.0728976
time: 0.31 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.40 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.40
Output dim: 0, lower bound: -0.0730615, upper bound: 0.0730888
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.40
Output dim: 0, lower bound: -0.0730615, upper bound: 0.0730888
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.40
Output dim: 0, lower bound: -0.0730615, upper bound: 0.0730615
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.40
Output dim: 0, lower bound: -0.0730615, upper bound: 0.0730615
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.40
Output dim: 0, lower bound: -0.0735916, upper bound: 0.0731125
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.40
Output dim: 0, lower bound: -0.0735845, upper bound: 0.0731103
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.40
Output dim: 0, lower bound: -0.0731094, upper bound: 0.0735058
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.40
Output dim: 0, lower bound: -0.0731114, upper bound: 0.0735532
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.40
Output dim: 0, lower bound: -0.0722489, upper bound: 0.0728976
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.40
Output dim: 0, lower bound: -0.0722400, upper bound: 0.0728976
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

Time for candidate selection: 0.00 seconds

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
- Time for RS candidates: 0.60 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.60
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753536
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.60
Output dim: 0, lower bound: -0.0753536, upper bound: 0.0753784

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

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

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0750927, upper bound: 0.0750934
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0750927, upper bound: 0.0750980
time: 0.29 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.39 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.39
Output dim: 0, lower bound: -0.0752233, upper bound: 0.0739373
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.39
Output dim: 0, lower bound: -0.0734268, upper bound: 0.0753228
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.39
Output dim: 0, lower bound: -0.0750927, upper bound: 0.0750934
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.39
Output dim: 0, lower bound: -0.0750927, upper bound: 0.0750980

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

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
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

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

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0733262
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
time: 0.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0733124
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750961
time: 0.29 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.43 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.43
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0736330
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.43
Output dim: 0, lower bound: -0.0733262, upper bound: 0.0734262
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750927
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0751000
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.43
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0733262
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0750934
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.43
Output dim: 0, lower bound: -0.0733124, upper bound: 0.0733124
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.43
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
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 15

Time for candidate selection: 0.00 seconds

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
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0730615, upper bound: 0.0730615
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0730615, upper bound: 0.0730615
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0722400, upper bound: 0.0723976
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0722400, upper bound: 0.0724417
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0731094, upper bound: 0.0731094
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0731184, upper bound: 0.0734931
time: 0.30 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.44 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749547
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0731368, upper bound: 0.0749518
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0730615, upper bound: 0.0730615
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0730615, upper bound: 0.0730615
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0722400, upper bound: 0.0723976
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0722400, upper bound: 0.0724417
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0731094, upper bound: 0.0731094
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0731184, upper bound: 0.0734931

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
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 31

Time for candidate selection: 0.00 seconds

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
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0729835, upper bound: 0.0742756
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0729835, upper bound: 0.0748030
time: 0.30 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.45 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.45
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0721131
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.45
Output dim: 0, lower bound: -0.0721101, upper bound: 0.0722143
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 0, lower bound: -0.0729835, upper bound: 0.0742756
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 0, lower bound: -0.0729835, upper bound: 0.0748030

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0727501, upper bound: 0.0741917
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727501, upper bound: 0.0727501
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727452, upper bound: 0.0727452
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0727452, upper bound: 0.0727452
time: 0.30 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.43 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.43
Output dim: 0, lower bound: -0.0727501, upper bound: 0.0741917
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.43
Output dim: 0, lower bound: -0.0727501, upper bound: 0.0727501
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.43
Output dim: 0, lower bound: -0.0727452, upper bound: 0.0727452
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.43
Output dim: 0, lower bound: -0.0727452, upper bound: 0.0727452

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144
1: -0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303
2: -0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905
3: -0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251
4: -0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0715515, upper bound: 0.0715597
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0715515, upper bound: 0.0715787
time: 0.30 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.40 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.40
Output dim: 0, lower bound: -0.0715515, upper bound: 0.0715597
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.40
Output dim: 0, lower bound: -0.0715515, upper bound: 0.0715787
Binary search (step 17): status=Status.VERIFIED, low=0.1818176, high=0.1818182, mid=0.1818176, abs_max=0.09181444346904755
rel_dist={0: [-0.07549650593465267, 0.07549650593465268]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.18181755448245165
execution time: 1026.46 seconds
