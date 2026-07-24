## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_1.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 0.6627565950000001


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017)
1: (-0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568)
2: (-0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153)
3: (-0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125)
4: (-0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177)

## BASE Result
execution time: IAR + LP analysis = 1.63 + 0.90 = 2.53 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.7126415, upper bound: 0.7126415


# Binary Search by BASE starts (time budget: 1197.47 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.0909091


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.0909091, mid=0.0909091, abs_max=0.7819017171859741
rel_dist={0: [-0.7126414684186158, 0.7126414684186156]}

## Binary search (step 1) starts
Candidate diff: 0.0454545


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0454545, mid=0.0454545, abs_max=0.7819017171859741
rel_dist={0: [-0.7025414092293646, 0.7025414092293644]}

## Binary search (step 2) starts
Candidate diff: 0.0227273


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0227273, mid=0.0227273, abs_max=0.7819017171859741
rel_dist={0: [-0.6961088071763791, 0.6961088071763795]}

## Binary search (step 3) starts
Candidate diff: 0.0113636


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0113636, mid=0.0113636, abs_max=0.7819017171859741
rel_dist={0: [-0.6921975474925997, 0.6921975474925999]}

## Binary search (step 4) starts
Candidate diff: 0.0056818


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0056818, mid=0.0056818, abs_max=0.7819017171859741
rel_dist={0: [-0.6884873820282374, 0.6884873820282374]}

## Binary search (step 5) starts
Candidate diff: 0.0028409


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0028409, mid=0.0028409, abs_max=0.7819017171859741
rel_dist={0: [-0.6721193804581553, 0.6721193804751224]}

## Binary search (step 6) starts
Candidate diff: 0.0014205


## IAR start
Binary search (step 6): status=Status.VERIFIED, low=0.0014205, high=0.0028409, mid=0.0014205, abs_max=0.7819017171859741
rel_dist={0: [-0.6600263838023774, 0.6600263837903424]}

## Binary search (step 7) starts
Candidate diff: 0.0021307


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0014205, high=0.0021307, mid=0.0021307, abs_max=0.7819017171859741
rel_dist={0: [-0.6662061409201548, 0.6662061409037836]}

## Binary search (step 8) starts
Candidate diff: 0.0017756


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0014205, high=0.0017756, mid=0.0017756, abs_max=0.7819017171859741
rel_dist={0: [-0.6631384487702139, 0.6631384487572083]}

## Binary search (step 9) starts
Candidate diff: 0.0015980


## IAR start
Binary search (step 9): status=Status.VERIFIED, low=0.0015980, high=0.0017756, mid=0.0015980, abs_max=0.7819017171859741
rel_dist={0: [-0.6615968384956282, 0.6615968384965103]}

## Binary search (step 10) starts
Candidate diff: 0.0016868


## IAR start
Binary search (step 10): status=Status.VERIFIED, low=0.0016868, high=0.0017756, mid=0.0016868, abs_max=0.7819017171859741
rel_dist={0: [-0.6623706919148379, 0.6623706919303145]}

## Binary search (step 11) starts
Candidate diff: 0.0017312


## IAR start
Binary search (step 11): status=Status.VERIFIED, low=0.0017312, high=0.0017756, mid=0.0017312, abs_max=0.7819017171859741
rel_dist={0: [-0.6627545709195036, 0.6627545709354368]}

## Binary search (step 12) starts
Candidate diff: 0.0017534


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0017312, high=0.0017534, mid=0.0017534, abs_max=0.7819017171859741
rel_dist={0: [-0.6629465097752838, 0.6629465097623557]}

## Binary search (step 13) starts
Candidate diff: 0.0017423


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0017312, high=0.0017423, mid=0.0017423, abs_max=0.7819017171859741
rel_dist={0: [-0.6628505402481032, 0.6628505402641487]}

## Binary search (step 14) starts
Candidate diff: 0.0017367


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0017312, high=0.0017367, mid=0.0017367, abs_max=0.7819017171859741
rel_dist={0: [-0.6628025550761784, 0.6628025550633088]}

## Binary search (step 15) starts
Candidate diff: 0.0017340


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0017312, high=0.0017340, mid=0.0017340, abs_max=0.7819017171859741
rel_dist={0: [-0.6627785629347587, 0.6627785629218987]}

## Binary search (step 16) starts
Candidate diff: 0.0017326


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0017312, high=0.0017326, mid=0.0017326, abs_max=0.7819017171859741
rel_dist={0: [-0.6627665663546163, 0.6627665663705633]}

## Binary search (step 17) starts
Candidate diff: 0.0017319


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0017312, high=0.0017319, mid=0.0017319, abs_max=0.7819017171859741
rel_dist={0: [-0.6627605681877892, 0.6627605682037285]}

## Binary Search Result
Binary search time: 46.07 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0017311790288658813


# Relational Split (RS_random_Z) starts
Time budget: 1151.40 seconds

## Binary search (step 0) starts
Candidate diff: 0.0917747


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6668734, upper bound: 0.6668734
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6668734, upper bound: 0.6668734
time: 0.33 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.68 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.68
Output dim: 0, lower bound: -0.6668734, upper bound: 0.6668734
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.68
Output dim: 0, lower bound: -0.6668734, upper bound: 0.6668734

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6666342, upper bound: 0.6456590
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6426757, upper bound: 0.6666342
time: 0.28 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6666342, upper bound: 0.6426757
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6456590, upper bound: 0.6666342
time: 0.29 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.11 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.11
Output dim: 0, lower bound: -0.6666342, upper bound: 0.6456590
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.11
Output dim: 0, lower bound: -0.6426757, upper bound: 0.6666342
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.11
Output dim: 0, lower bound: -0.6666342, upper bound: 0.6426757
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.11
Output dim: 0, lower bound: -0.6456590, upper bound: 0.6666342

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6472159, upper bound: 0.6444362
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6430371, upper bound: 0.6456590
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6366587, upper bound: 0.6408665
time: 0.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6366932, upper bound: 0.6408665
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6366932
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6366587
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6456590, upper bound: 0.6430371
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6444362, upper bound: 0.6472159
time: 0.29 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.21 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.21
Output dim: 0, lower bound: -0.6472159, upper bound: 0.6444362
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.21
Output dim: 0, lower bound: -0.6430371, upper bound: 0.6456590
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.21
Output dim: 0, lower bound: -0.6366587, upper bound: 0.6408665
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.21
Output dim: 0, lower bound: -0.6366932, upper bound: 0.6408665
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.21
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6366932
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.21
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6366587
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.21
Output dim: 0, lower bound: -0.6456590, upper bound: 0.6430371
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.21
Output dim: 0, lower bound: -0.6444362, upper bound: 0.6472159
Binary search (step 0): status=Status.VERIFIED, low=0.0917747, high=0.1818182, mid=0.0917747, abs_max=0.7819017171859741
rel_dist={0: [-0.7126414684186158, 0.7126414684186155]}

## Binary search (step 1) starts
Candidate diff: 0.1367964


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6625625, upper bound: 0.6625625
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6625625, upper bound: 0.6625625
time: 0.32 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.64 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 0.64
Output dim: 0, lower bound: -0.6625625, upper bound: 0.6625625
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 0.64
Output dim: 0, lower bound: -0.6625625, upper bound: 0.6625625
Binary search (step 1): status=Status.VERIFIED, low=0.1367964, high=0.1818182, mid=0.1367964, abs_max=0.7819017171859741
rel_dist={0: [-0.7126414684186158, 0.7126414684186158]}

## Binary search (step 2) starts
Candidate diff: 0.1593073


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6778737, upper bound: 0.6778737
time: 0.27 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6778737, upper bound: 0.6778737
time: 0.27 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.56 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.56
Output dim: 0, lower bound: -0.6778737, upper bound: 0.6778737
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.56
Output dim: 0, lower bound: -0.6778737, upper bound: 0.6778737

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6590587, upper bound: 0.6772592
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6772592, upper bound: 0.6533867
time: 0.32 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6430371, upper bound: 0.6484521
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6444363, upper bound: 0.6472159
time: 0.30 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.10 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.10
Output dim: 0, lower bound: -0.6590587, upper bound: 0.6772592
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.10
Output dim: 0, lower bound: -0.6772592, upper bound: 0.6533867
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.10
Output dim: 0, lower bound: -0.6430371, upper bound: 0.6484521
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.10
Output dim: 0, lower bound: -0.6444363, upper bound: 0.6472159

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6472159, upper bound: 0.6444362
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6484521, upper bound: 0.6425515
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6426757, upper bound: 0.6443244
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6456590, upper bound: 0.6430371
time: 0.31 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.16 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.16
Output dim: 0, lower bound: -0.6472159, upper bound: 0.6444362
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.16
Output dim: 0, lower bound: -0.6484521, upper bound: 0.6425515
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.16
Output dim: 0, lower bound: -0.6426757, upper bound: 0.6443244
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.16
Output dim: 0, lower bound: -0.6456590, upper bound: 0.6430371
Binary search (step 2): status=Status.VERIFIED, low=0.1593073, high=0.1818182, mid=0.1593073, abs_max=0.7819017171859741
rel_dist={0: [-0.7126414684186158, 0.7126414684186158]}

## Binary search (step 3) starts
Candidate diff: 0.1705627


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6625625, upper bound: 0.6625625
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6625625, upper bound: 0.6625625
time: 0.29 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.60 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 0.60
Output dim: 0, lower bound: -0.6625625, upper bound: 0.6625625
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 0.60
Output dim: 0, lower bound: -0.6625625, upper bound: 0.6625625
Binary search (step 3): status=Status.VERIFIED, low=0.1705627, high=0.1818182, mid=0.1705627, abs_max=0.7819017171859741
rel_dist={0: [-0.7126414684186158, 0.7126414684186158]}

## Binary search (step 4) starts
Candidate diff: 0.1761905


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7053783, upper bound: 0.7051275
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7051275, upper bound: 0.7053783
time: 0.27 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.59 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.59
Output dim: 0, lower bound: -0.7053783, upper bound: 0.7051275
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.59
Output dim: 0, lower bound: -0.7051275, upper bound: 0.7053783

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7039676, upper bound: 0.6948314
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6905401, upper bound: 0.7037169
time: 0.30 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7037169, upper bound: 0.6905401
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6948314, upper bound: 0.7039676
time: 0.29 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.12 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.12
Output dim: 0, lower bound: -0.7039676, upper bound: 0.6948314
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.12
Output dim: 0, lower bound: -0.6905401, upper bound: 0.7037169
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.12
Output dim: 0, lower bound: -0.7037169, upper bound: 0.6905401
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.12
Output dim: 0, lower bound: -0.6948314, upper bound: 0.7039676

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6605439, upper bound: 0.6619682
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6576264, upper bound: 0.6619682
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6605849, upper bound: 0.6605157
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6600525, upper bound: 0.6613721
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6613721, upper bound: 0.6600525
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6605157, upper bound: 0.6605849
time: 0.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6576264
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6605439
time: 0.29 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.17 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.17
Output dim: 0, lower bound: -0.6605439, upper bound: 0.6619682
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.17
Output dim: 0, lower bound: -0.6576264, upper bound: 0.6619682
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.17
Output dim: 0, lower bound: -0.6605849, upper bound: 0.6605157
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.17
Output dim: 0, lower bound: -0.6600525, upper bound: 0.6613721
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.17
Output dim: 0, lower bound: -0.6613721, upper bound: 0.6600525
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.17
Output dim: 0, lower bound: -0.6605157, upper bound: 0.6605849
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.17
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6576264
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.17
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6605439
Binary search (step 4): status=Status.VERIFIED, low=0.1761905, high=0.1818182, mid=0.1761905, abs_max=0.7819017171859741
rel_dist={0: [-0.7126414684186158, 0.7126414685280116]}

## Binary search (step 5) starts
Candidate diff: 0.1790043


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7008613, upper bound: 0.7008613
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7008613, upper bound: 0.7092829
time: 0.30 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.65 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.65
Output dim: 0, lower bound: -0.7008613, upper bound: 0.7008613
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.65
Output dim: 0, lower bound: -0.7008613, upper bound: 0.7092829

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6737222, upper bound: 0.6531202
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6737222, upper bound: 0.6537564
time: 0.28 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6537564, upper bound: 0.6737222
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6531202, upper bound: 0.6737222
time: 0.28 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.12 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.12
Output dim: 0, lower bound: -0.6737222, upper bound: 0.6531202
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.12
Output dim: 0, lower bound: -0.6737222, upper bound: 0.6537564
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.12
Output dim: 0, lower bound: -0.6537564, upper bound: 0.6737222
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.12
Output dim: 0, lower bound: -0.6531202, upper bound: 0.6737222

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6559504, upper bound: 0.6527017
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6731077, upper bound: 0.6496885
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6504372, upper bound: 0.6534141
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6731077, upper bound: 0.6535135
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6535135, upper bound: 0.6731077
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6534141, upper bound: 0.6504372
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6368080, upper bound: 0.6408665
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6370055, upper bound: 0.6408665
time: 0.31 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.85 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.85
Output dim: 0, lower bound: -0.6559504, upper bound: 0.6527017
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.85
Output dim: 0, lower bound: -0.6731077, upper bound: 0.6496885
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.85
Output dim: 0, lower bound: -0.6504372, upper bound: 0.6534141
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.85
Output dim: 0, lower bound: -0.6731077, upper bound: 0.6535135
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.85
Output dim: 0, lower bound: -0.6535135, upper bound: 0.6731077
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.85
Output dim: 0, lower bound: -0.6534141, upper bound: 0.6504372
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.85
Output dim: 0, lower bound: -0.6368080, upper bound: 0.6408665
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.85
Output dim: 0, lower bound: -0.6370055, upper bound: 0.6408665

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6366587, upper bound: 0.6369197
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6371619, upper bound: 0.6368080
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6366587, upper bound: 0.6408665
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6370055, upper bound: 0.6408665
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6368080, upper bound: 0.6370055
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6366587
time: 0.29 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.50 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.50
Output dim: 0, lower bound: -0.6366587, upper bound: 0.6369197
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.50
Output dim: 0, lower bound: -0.6371619, upper bound: 0.6368080
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.50
Output dim: 0, lower bound: -0.6366587, upper bound: 0.6408665
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.50
Output dim: 0, lower bound: -0.6370055, upper bound: 0.6408665
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.50
Output dim: 0, lower bound: -0.6368080, upper bound: 0.6370055
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.50
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6366587
Binary search (step 5): status=Status.VERIFIED, low=0.1790043, high=0.1818182, mid=0.1790043, abs_max=0.7819017171859741
rel_dist={0: [-0.7126414684186158, 0.7126414685280116]}

## Binary search (step 6) starts
Candidate diff: 0.1804113


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6668734, upper bound: 0.6668734
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6668734, upper bound: 0.6668734
time: 0.33 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.70 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.70
Output dim: 0, lower bound: -0.6668734, upper bound: 0.6668734
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.70
Output dim: 0, lower bound: -0.6668734, upper bound: 0.6668734

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6408665
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6408665
time: 0.27 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6408665
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6408665
time: 0.27 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.45 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.45
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6408665
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.45
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6408665
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.45
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6408665
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.45
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6408665
Binary search (step 6): status=Status.VERIFIED, low=0.1804113, high=0.1818182, mid=0.1804113, abs_max=0.7819017171859741
rel_dist={0: [-0.7126414684186158, 0.7126414685280116]}

## Binary search (step 7) starts
Candidate diff: 0.1811147


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6778737, upper bound: 0.6778737
time: 0.27 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6778737, upper bound: 0.6778737
time: 0.27 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.57 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.57
Output dim: 0, lower bound: -0.6778737, upper bound: 0.6778737
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.57
Output dim: 0, lower bound: -0.6778737, upper bound: 0.6778737

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6737222, upper bound: 0.6531202
time: 0.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6537564, upper bound: 0.6737222
time: 0.31 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6430371, upper bound: 0.6484521
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6444363, upper bound: 0.6472159
time: 0.31 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.11 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.11
Output dim: 0, lower bound: -0.6737222, upper bound: 0.6531202
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.11
Output dim: 0, lower bound: -0.6537564, upper bound: 0.6737222
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.11
Output dim: 0, lower bound: -0.6430371, upper bound: 0.6484521
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.11
Output dim: 0, lower bound: -0.6444363, upper bound: 0.6472159

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6559504, upper bound: 0.6527017
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6731077, upper bound: 0.6496885
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6535135, upper bound: 0.6731077
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6534141, upper bound: 0.6504372
time: 0.29 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.16 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.16
Output dim: 0, lower bound: -0.6559504, upper bound: 0.6527017
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.16
Output dim: 0, lower bound: -0.6731077, upper bound: 0.6496885
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.16
Output dim: 0, lower bound: -0.6535135, upper bound: 0.6731077
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.16
Output dim: 0, lower bound: -0.6534141, upper bound: 0.6504372

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6366587, upper bound: 0.6369197
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6371619, upper bound: 0.6368080
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6370055
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6366587
time: 0.29 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.80 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.80
Output dim: 0, lower bound: -0.6366587, upper bound: 0.6369197
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.80
Output dim: 0, lower bound: -0.6371619, upper bound: 0.6368080
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.80
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6370055
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.80
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6366587
Binary search (step 7): status=Status.VERIFIED, low=0.1811147, high=0.1818182, mid=0.1811147, abs_max=0.7819017171859741
rel_dist={0: [-0.7126414684186158, 0.7126414685280116]}

## Binary search (step 8) starts
Candidate diff: 0.1814665


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6625625, upper bound: 0.6625625
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6625625, upper bound: 0.6625625
time: 0.28 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.60 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 0.60
Output dim: 0, lower bound: -0.6625625, upper bound: 0.6625625
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 0.60
Output dim: 0, lower bound: -0.6625625, upper bound: 0.6625625
Binary search (step 8): status=Status.VERIFIED, low=0.1814665, high=0.1818182, mid=0.1814665, abs_max=0.7819017171859741
rel_dist={0: [-0.7126414684186158, 0.7126414685280116]}

## Binary search (step 9) starts
Candidate diff: 0.1816423


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6625625, upper bound: 0.6625625
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6625625, upper bound: 0.6625625
time: 0.28 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.59 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 0.59
Output dim: 0, lower bound: -0.6625625, upper bound: 0.6625625
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 0.59
Output dim: 0, lower bound: -0.6625625, upper bound: 0.6625625
Binary search (step 9): status=Status.VERIFIED, low=0.1816423, high=0.1818182, mid=0.1816423, abs_max=0.7819017171859741
rel_dist={0: [-0.7126414684186158, 0.7126414685280116]}

## Binary search (step 10) starts
Candidate diff: 0.1817303


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7092829, upper bound: 0.7008613
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7008613, upper bound: 0.7092829
time: 0.28 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.61 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.61
Output dim: 0, lower bound: -0.7092829, upper bound: 0.7008613
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.61
Output dim: 0, lower bound: -0.7008613, upper bound: 0.7092829

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6737222, upper bound: 0.6531202
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6531202, upper bound: 0.6537564
time: 0.30 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6537564, upper bound: 0.6737222
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6531202, upper bound: 0.6737222
time: 0.28 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.12 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.12
Output dim: 0, lower bound: -0.6737222, upper bound: 0.6531202
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.12
Output dim: 0, lower bound: -0.6531202, upper bound: 0.6537564
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.12
Output dim: 0, lower bound: -0.6537564, upper bound: 0.6737222
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.12
Output dim: 0, lower bound: -0.6531202, upper bound: 0.6737222

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6370055
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6368080
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6535135, upper bound: 0.6731077
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6534141, upper bound: 0.6504372
time: 0.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6368080, upper bound: 0.6408665
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6370055, upper bound: 0.6408665
time: 0.30 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.15 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.15
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6370055
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.15
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6368080
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.15
Output dim: 0, lower bound: -0.6535135, upper bound: 0.6731077
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.15
Output dim: 0, lower bound: -0.6534141, upper bound: 0.6504372
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.15
Output dim: 0, lower bound: -0.6368080, upper bound: 0.6408665
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.15
Output dim: 0, lower bound: -0.6370055, upper bound: 0.6408665

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6370055
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6366587
time: 0.29 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.81 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.81
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6370055
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.81
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6366587
Binary search (step 10): status=Status.VERIFIED, low=0.1817303, high=0.1818182, mid=0.1817303, abs_max=0.7819017171859741
rel_dist={0: [-0.7126414684186158, 0.7126414685280116]}

## Binary search (step 11) starts
Candidate diff: 0.1817742


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6625625, upper bound: 0.6625625
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6625625, upper bound: 0.6625625
time: 0.30 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.62 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 0.62
Output dim: 0, lower bound: -0.6625625, upper bound: 0.6625625
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 0.62
Output dim: 0, lower bound: -0.6625625, upper bound: 0.6625625
Binary search (step 11): status=Status.VERIFIED, low=0.1817742, high=0.1818182, mid=0.1817742, abs_max=0.7819017171859741
rel_dist={0: [-0.7126414684186158, 0.7126414685280116]}

## Binary search (step 12) starts
Candidate diff: 0.1817962


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7092829, upper bound: 0.7008613
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7008613, upper bound: 0.7092829
time: 0.28 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.61 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.61
Output dim: 0, lower bound: -0.7092829, upper bound: 0.7008613
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.61
Output dim: 0, lower bound: -0.7008613, upper bound: 0.7092829

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6617165, upper bound: 0.6623656
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6609033, upper bound: 0.6623656
time: 0.28 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6888340, upper bound: 0.7067790
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6833647, upper bound: 0.7066308
time: 0.30 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.15 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.15
Output dim: 0, lower bound: -0.6617165, upper bound: 0.6623656
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.15
Output dim: 0, lower bound: -0.6609033, upper bound: 0.6623656
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.15
Output dim: 0, lower bound: -0.6888340, upper bound: 0.7067790
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.15
Output dim: 0, lower bound: -0.6833647, upper bound: 0.7066308

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6371619
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6366587
time: 0.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6366932, upper bound: 0.6408665
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6371619, upper bound: 0.6408665
time: 0.29 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.45 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.45
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6371619
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.45
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6366587
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.45
Output dim: 0, lower bound: -0.6366932, upper bound: 0.6408665
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.45
Output dim: 0, lower bound: -0.6371619, upper bound: 0.6408665
Binary search (step 12): status=Status.VERIFIED, low=0.1817962, high=0.1818182, mid=0.1817962, abs_max=0.7819017171859741
rel_dist={0: [-0.7126414684186158, 0.7126414685280116]}

## Binary search (step 13) starts
Candidate diff: 0.1818072


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6668734, upper bound: 0.6668734
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6668734, upper bound: 0.6668734
time: 0.31 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.66 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.66
Output dim: 0, lower bound: -0.6668734, upper bound: 0.6668734
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.66
Output dim: 0, lower bound: -0.6668734, upper bound: 0.6668734

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6472159, upper bound: 0.6444363
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6430371, upper bound: 0.6484521
time: 0.27 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6666342, upper bound: 0.6426757
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6456590, upper bound: 0.6666342
time: 0.29 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.09 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.09
Output dim: 0, lower bound: -0.6472159, upper bound: 0.6444363
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.09
Output dim: 0, lower bound: -0.6430371, upper bound: 0.6484521
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.09
Output dim: 0, lower bound: -0.6666342, upper bound: 0.6426757
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.09
Output dim: 0, lower bound: -0.6456590, upper bound: 0.6666342

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6366932
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6366587
time: 0.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6371619, upper bound: 0.6408665
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6371619, upper bound: 0.6408665
time: 0.30 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.12 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6366932
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6366587
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -0.6371619, upper bound: 0.6408665
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -0.6371619, upper bound: 0.6408665
Binary search (step 13): status=Status.VERIFIED, low=0.1818072, high=0.1818182, mid=0.1818072, abs_max=0.7819017171859741
rel_dist={0: [-0.7126414684186158, 0.7126414685280116]}

## Binary search (step 14) starts
Candidate diff: 0.1818127


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6625625, upper bound: 0.6625625
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6625625, upper bound: 0.6625625
time: 0.28 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.60 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 0.60
Output dim: 0, lower bound: -0.6625625, upper bound: 0.6625625
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 0.60
Output dim: 0, lower bound: -0.6625625, upper bound: 0.6625625
Binary search (step 14): status=Status.VERIFIED, low=0.1818127, high=0.1818182, mid=0.1818127, abs_max=0.7819017171859741
rel_dist={0: [-0.7126414684186158, 0.7126414685280116]}

## Binary search (step 15) starts
Candidate diff: 0.1818154


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7008613, upper bound: 0.7008613
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7008613, upper bound: 0.7092829
time: 0.27 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.63 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.63
Output dim: 0, lower bound: -0.7008613, upper bound: 0.7008613
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.63
Output dim: 0, lower bound: -0.7008613, upper bound: 0.7092829

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7066308, upper bound: 0.6833647
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7067790, upper bound: 0.6888340
time: 0.28 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6408665
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6408665
time: 0.31 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.15 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.15
Output dim: 0, lower bound: -0.7066308, upper bound: 0.6833647
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.15
Output dim: 0, lower bound: -0.7067790, upper bound: 0.6888340
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.15
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6408665
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.15
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6408665

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6559504, upper bound: 0.6527017
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6504372, upper bound: 0.6534141
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6731077, upper bound: 0.6496885
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6731077, upper bound: 0.6535135
time: 0.29 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.49 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.49
Output dim: 0, lower bound: -0.6559504, upper bound: 0.6527017
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.49
Output dim: 0, lower bound: -0.6504372, upper bound: 0.6534141
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.49
Output dim: 0, lower bound: -0.6731077, upper bound: 0.6496885
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.49
Output dim: 0, lower bound: -0.6731077, upper bound: 0.6535135

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6366587, upper bound: 0.6369197
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6371619, upper bound: 0.6368080
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6366587, upper bound: 0.6408665
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6370055, upper bound: 0.6408665
time: 0.31 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.76 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.76
Output dim: 0, lower bound: -0.6366587, upper bound: 0.6369197
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.76
Output dim: 0, lower bound: -0.6371619, upper bound: 0.6368080
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.76
Output dim: 0, lower bound: -0.6366587, upper bound: 0.6408665
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.76
Output dim: 0, lower bound: -0.6370055, upper bound: 0.6408665
Binary search (step 15): status=Status.VERIFIED, low=0.1818154, high=0.1818182, mid=0.1818154, abs_max=0.7819017171859741
rel_dist={0: [-0.7126414684186158, 0.7126414685280116]}

## Binary search (step 16) starts
Candidate diff: 0.1818168


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6778737, upper bound: 0.6778737
time: 0.27 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6778737, upper bound: 0.6778737
time: 0.28 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.57 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.57
Output dim: 0, lower bound: -0.6778737, upper bound: 0.6778737
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.57
Output dim: 0, lower bound: -0.6778737, upper bound: 0.6778737

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6590587, upper bound: 0.6772592
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6772592, upper bound: 0.6533867
time: 0.31 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6430371, upper bound: 0.6484521
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6444363, upper bound: 0.6472159
time: 0.31 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.10 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.10
Output dim: 0, lower bound: -0.6590587, upper bound: 0.6772592
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.10
Output dim: 0, lower bound: -0.6772592, upper bound: 0.6533867
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.10
Output dim: 0, lower bound: -0.6430371, upper bound: 0.6484521
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.10
Output dim: 0, lower bound: -0.6444363, upper bound: 0.6472159

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6472159, upper bound: 0.6444363
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6484521, upper bound: 0.6425515
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6426757, upper bound: 0.6443244
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6456590, upper bound: 0.6430371
time: 0.31 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.17 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.17
Output dim: 0, lower bound: -0.6472159, upper bound: 0.6444363
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.17
Output dim: 0, lower bound: -0.6484521, upper bound: 0.6425515
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.17
Output dim: 0, lower bound: -0.6426757, upper bound: 0.6443244
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.17
Output dim: 0, lower bound: -0.6456590, upper bound: 0.6430371
Binary search (step 16): status=Status.VERIFIED, low=0.1818168, high=0.1818182, mid=0.1818168, abs_max=0.7819017171859741
rel_dist={0: [-0.7126414684186158, 0.7126414684186158]}

## Binary search (step 17) starts
Candidate diff: 0.1818175


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7100758, upper bound: 0.7102848
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7100758, upper bound: 0.7100758
time: 0.29 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.59 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.59
Output dim: 0, lower bound: -0.7100758, upper bound: 0.7102848
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.59
Output dim: 0, lower bound: -0.7100758, upper bound: 0.7100758

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7066308, upper bound: 0.6833647
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6888340, upper bound: 0.7067790
time: 0.30 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6772592, upper bound: 0.6533867
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6772592, upper bound: 0.6590587
time: 0.30 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.86 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.86
Output dim: 0, lower bound: -0.7066308, upper bound: 0.6833647
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.86
Output dim: 0, lower bound: -0.6888340, upper bound: 0.7067790
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.86
Output dim: 0, lower bound: -0.6772592, upper bound: 0.6533867
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.86
Output dim: 0, lower bound: -0.6772592, upper bound: 0.6590587

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6371619
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6366932
time: 0.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6371619
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6366587
time: 0.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6731077, upper bound: 0.6496885
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6534141, upper bound: 0.6504372
time: 0.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6731077, upper bound: 0.6535135
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6527017, upper bound: 0.6559504
time: 0.31 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.53 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.53
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6371619
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.53
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6366932
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.53
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6371619
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.53
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6366587
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -0.6731077, upper bound: 0.6496885
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.53
Output dim: 0, lower bound: -0.6534141, upper bound: 0.6504372
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -0.6731077, upper bound: 0.6535135
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.53
Output dim: 0, lower bound: -0.6527017, upper bound: 0.6559504

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6366587, upper bound: 0.6369197
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6371619, upper bound: 0.6368080
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6366587, upper bound: 0.6408665
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6370055, upper bound: 0.6408665
time: 0.31 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.81 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.81
Output dim: 0, lower bound: -0.6366587, upper bound: 0.6369197
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.81
Output dim: 0, lower bound: -0.6371619, upper bound: 0.6368080
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.81
Output dim: 0, lower bound: -0.6366587, upper bound: 0.6408665
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.81
Output dim: 0, lower bound: -0.6370055, upper bound: 0.6408665
Binary search (step 17): status=Status.VERIFIED, low=0.1818175, high=0.1818182, mid=0.1818175, abs_max=0.7819017171859741
rel_dist={0: [-0.7126414684186158, 0.7126414685280116]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.1818175002593681
execution time: 211.20 seconds
