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
execution time: IAR + LP analysis = 1.62 + 0.91 = 2.54 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.7126415, upper bound: 0.7126415


# Binary Search by BASE starts (time budget: 1197.46 seconds, max iter: 100)

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
Binary search time: 46.45 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0017311790288658813


# Relational Split (RS_dual_Z) starts
Time budget: 1151.01 seconds

## Binary search (step 0) starts
Candidate diff: 0.0917747


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7053783, upper bound: 0.7051275
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7051275, upper bound: 0.7053783
time: 0.30 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.74 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.74
Output dim: 0, lower bound: -0.7053783, upper bound: 0.7051275
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.74
Output dim: 0, lower bound: -0.7051275, upper bound: 0.7053783

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7039676, upper bound: 0.6940492
time: 0.31 seconds

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

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7037169, upper bound: 0.6905401
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6940492, upper bound: 0.7039676
time: 0.36 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.33 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.33
Output dim: 0, lower bound: -0.7039676, upper bound: 0.6940492
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.33
Output dim: 0, lower bound: -0.6905401, upper bound: 0.7037169
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.33
Output dim: 0, lower bound: -0.7037169, upper bound: 0.6905401
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.33
Output dim: 0, lower bound: -0.6940492, upper bound: 0.7039676

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6605439, upper bound: 0.6619682
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6576264, upper bound: 0.6619682
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6605849, upper bound: 0.6605157
time: 0.32 seconds

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

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6613721, upper bound: 0.6600525
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6605157, upper bound: 0.6605849
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

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
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6605439
time: 0.32 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.04 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.04
Output dim: 0, lower bound: -0.6605439, upper bound: 0.6619682
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.04
Output dim: 0, lower bound: -0.6576264, upper bound: 0.6619682
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.04
Output dim: 0, lower bound: -0.6605849, upper bound: 0.6605157
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.04
Output dim: 0, lower bound: -0.6600525, upper bound: 0.6613721
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.04
Output dim: 0, lower bound: -0.6613721, upper bound: 0.6600525
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.04
Output dim: 0, lower bound: -0.6605157, upper bound: 0.6605849
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.04
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6576264
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.04
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6605439
Binary search (step 0): status=Status.VERIFIED, low=0.0917747, high=0.1818182, mid=0.0917747, abs_max=0.7819017171859741
rel_dist={0: [-0.7126414684186158, 0.7126414684186155]}

## Binary search (step 1) starts
Candidate diff: 0.1367964


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

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
time: 0.28 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.73 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.73
Output dim: 0, lower bound: -0.7053783, upper bound: 0.7051275
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.73
Output dim: 0, lower bound: -0.7051275, upper bound: 0.7053783

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7039676, upper bound: 0.6948314
time: 0.33 seconds

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

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7037169, upper bound: 0.6905401
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6948314, upper bound: 0.7039676
time: 0.29 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.31 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 0, lower bound: -0.7039676, upper bound: 0.6948314
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 0, lower bound: -0.6905401, upper bound: 0.7037169
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 0, lower bound: -0.7037169, upper bound: 0.6905401
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 0, lower bound: -0.6948314, upper bound: 0.7039676

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

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

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6605849, upper bound: 0.6605157
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6600525, upper bound: 0.6613721
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6613721, upper bound: 0.6600525
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6605157, upper bound: 0.6605849
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

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
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6605439
time: 0.29 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.23 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.23
Output dim: 0, lower bound: -0.6605439, upper bound: 0.6619682
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.23
Output dim: 0, lower bound: -0.6576264, upper bound: 0.6619682
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.23
Output dim: 0, lower bound: -0.6605849, upper bound: 0.6605157
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.23
Output dim: 0, lower bound: -0.6600525, upper bound: 0.6613721
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.23
Output dim: 0, lower bound: -0.6613721, upper bound: 0.6600525
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.23
Output dim: 0, lower bound: -0.6605157, upper bound: 0.6605849
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.23
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6576264
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.23
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6605439
Binary search (step 1): status=Status.VERIFIED, low=0.1367964, high=0.1818182, mid=0.1367964, abs_max=0.7819017171859741
rel_dist={0: [-0.7126414684186158, 0.7126414684186158]}

## Binary search (step 2) starts
Candidate diff: 0.1593073


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7051275, upper bound: 0.7051275
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7051275, upper bound: 0.7053783
time: 0.28 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.72 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.72
Output dim: 0, lower bound: -0.7051275, upper bound: 0.7051275
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.72
Output dim: 0, lower bound: -0.7051275, upper bound: 0.7053783

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7039676, upper bound: 0.6948314
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6905401, upper bound: 0.7037169
time: 0.29 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

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
time: 0.30 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.28 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.28
Output dim: 0, lower bound: -0.7039676, upper bound: 0.6948314
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.28
Output dim: 0, lower bound: -0.6905401, upper bound: 0.7037169
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.28
Output dim: 0, lower bound: -0.7037169, upper bound: 0.6905401
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.28
Output dim: 0, lower bound: -0.6948314, upper bound: 0.7039676

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6605439, upper bound: 0.6619682
time: 0.33 seconds

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

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6605849, upper bound: 0.6605157
time: 0.29 seconds

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

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6613721, upper bound: 0.6600525
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6605157, upper bound: 0.6605849
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

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
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6605439
time: 0.31 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.03 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.03
Output dim: 0, lower bound: -0.6605439, upper bound: 0.6619682
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.03
Output dim: 0, lower bound: -0.6576264, upper bound: 0.6619682
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.03
Output dim: 0, lower bound: -0.6605849, upper bound: 0.6605157
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.03
Output dim: 0, lower bound: -0.6600525, upper bound: 0.6613721
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.03
Output dim: 0, lower bound: -0.6613721, upper bound: 0.6600525
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.03
Output dim: 0, lower bound: -0.6605157, upper bound: 0.6605849
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.03
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6576264
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.03
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6605439
Binary search (step 2): status=Status.VERIFIED, low=0.1593073, high=0.1818182, mid=0.1593073, abs_max=0.7819017171859741
rel_dist={0: [-0.7126414684186158, 0.7126414684186158]}

## Binary search (step 3) starts
Candidate diff: 0.1705627


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7053783, upper bound: 0.7051275
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7051275, upper bound: 0.7053783
time: 0.27 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.72 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.72
Output dim: 0, lower bound: -0.7053783, upper bound: 0.7051275
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.72
Output dim: 0, lower bound: -0.7051275, upper bound: 0.7053783

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7039676, upper bound: 0.6948314
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6905401, upper bound: 0.7037169
time: 0.29 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

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
time: 0.31 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.30 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.30
Output dim: 0, lower bound: -0.7039676, upper bound: 0.6948314
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.30
Output dim: 0, lower bound: -0.6905401, upper bound: 0.7037169
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.30
Output dim: 0, lower bound: -0.7037169, upper bound: 0.6905401
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.30
Output dim: 0, lower bound: -0.6948314, upper bound: 0.7039676

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6605439, upper bound: 0.6619682
time: 0.32 seconds

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

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6605849, upper bound: 0.6605157
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6600525, upper bound: 0.6613721
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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6613721, upper bound: 0.6600525
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6605157, upper bound: 0.6605849
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

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
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6605439
time: 0.30 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.05 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.05
Output dim: 0, lower bound: -0.6605439, upper bound: 0.6619682
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.05
Output dim: 0, lower bound: -0.6576264, upper bound: 0.6619682
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.05
Output dim: 0, lower bound: -0.6605849, upper bound: 0.6605157
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.05
Output dim: 0, lower bound: -0.6600525, upper bound: 0.6613721
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.05
Output dim: 0, lower bound: -0.6613721, upper bound: 0.6600525
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.05
Output dim: 0, lower bound: -0.6605157, upper bound: 0.6605849
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.05
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6576264
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.05
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6605439
Binary search (step 3): status=Status.VERIFIED, low=0.1705627, high=0.1818182, mid=0.1705627, abs_max=0.7819017171859741
rel_dist={0: [-0.7126414684186158, 0.7126414684186158]}

## Binary search (step 4) starts
Candidate diff: 0.1761905


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7053783, upper bound: 0.7051275
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7051275, upper bound: 0.7053783
time: 0.29 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.78 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.78
Output dim: 0, lower bound: -0.7053783, upper bound: 0.7051275
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.78
Output dim: 0, lower bound: -0.7051275, upper bound: 0.7053783

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
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7039676, upper bound: 0.6948314
time: 0.34 seconds

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

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

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
time: 0.30 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.27 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.27
Output dim: 0, lower bound: -0.7039676, upper bound: 0.6948314
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.27
Output dim: 0, lower bound: -0.6905401, upper bound: 0.7037169
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.27
Output dim: 0, lower bound: -0.7037169, upper bound: 0.6905401
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.27
Output dim: 0, lower bound: -0.6948314, upper bound: 0.7039676

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6605439, upper bound: 0.6619682
time: 0.32 seconds

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

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6605849, upper bound: 0.6605157
time: 0.29 seconds

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

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

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
time: 0.29 seconds

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
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

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
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6605439
time: 0.29 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.00 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.00
Output dim: 0, lower bound: -0.6605439, upper bound: 0.6619682
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.00
Output dim: 0, lower bound: -0.6576264, upper bound: 0.6619682
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.00
Output dim: 0, lower bound: -0.6605849, upper bound: 0.6605157
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.00
Output dim: 0, lower bound: -0.6600525, upper bound: 0.6613721
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.00
Output dim: 0, lower bound: -0.6613721, upper bound: 0.6600525
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.00
Output dim: 0, lower bound: -0.6605157, upper bound: 0.6605849
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.00
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6576264
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.00
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6605439
Binary search (step 4): status=Status.VERIFIED, low=0.1761905, high=0.1818182, mid=0.1761905, abs_max=0.7819017171859741
rel_dist={0: [-0.7126414684186158, 0.7126414685280116]}

## Binary search (step 5) starts
Candidate diff: 0.1790043


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

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
time: 0.28 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.73 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.73
Output dim: 0, lower bound: -0.7053783, upper bound: 0.7051275
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.73
Output dim: 0, lower bound: -0.7051275, upper bound: 0.7053783

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
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

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

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7037169, upper bound: 0.6905401
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6948314, upper bound: 0.7039676
time: 0.29 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.49 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.49
Output dim: 0, lower bound: -0.7039676, upper bound: 0.6948314
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.49
Output dim: 0, lower bound: -0.6905401, upper bound: 0.7037169
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.49
Output dim: 0, lower bound: -0.7037169, upper bound: 0.6905401
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.49
Output dim: 0, lower bound: -0.6948314, upper bound: 0.7039676

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6605849, upper bound: 0.6605157
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6600525, upper bound: 0.6613721
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6613721, upper bound: 0.6600525
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6605157, upper bound: 0.6605849
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

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
- Time for RS candidates: 3.21 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.21
Output dim: 0, lower bound: -0.6605439, upper bound: 0.6619682
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.21
Output dim: 0, lower bound: -0.6576264, upper bound: 0.6619682
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.21
Output dim: 0, lower bound: -0.6605849, upper bound: 0.6605157
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.21
Output dim: 0, lower bound: -0.6600525, upper bound: 0.6613721
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.21
Output dim: 0, lower bound: -0.6613721, upper bound: 0.6600525
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.21
Output dim: 0, lower bound: -0.6605157, upper bound: 0.6605849
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.21
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6576264
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.21
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6605439
Binary search (step 5): status=Status.VERIFIED, low=0.1790043, high=0.1818182, mid=0.1790043, abs_max=0.7819017171859741
rel_dist={0: [-0.7126414684186158, 0.7126414685280116]}

## Binary search (step 6) starts
Candidate diff: 0.1804113


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7053783, upper bound: 0.7051275
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7051275, upper bound: 0.7053783
time: 0.26 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.70 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.70
Output dim: 0, lower bound: -0.7053783, upper bound: 0.7051275
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.70
Output dim: 0, lower bound: -0.7051275, upper bound: 0.7053783

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7039676, upper bound: 0.6948314
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6905401, upper bound: 0.7037169
time: 0.29 seconds

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7037169, upper bound: 0.6905401
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6948314, upper bound: 0.7039676
time: 0.31 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.28 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.28
Output dim: 0, lower bound: -0.7039676, upper bound: 0.6948314
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.28
Output dim: 0, lower bound: -0.6905401, upper bound: 0.7037169
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.28
Output dim: 0, lower bound: -0.7037169, upper bound: 0.6905401
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.28
Output dim: 0, lower bound: -0.6948314, upper bound: 0.7039676

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6605439, upper bound: 0.6619682
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6576264, upper bound: 0.6619682
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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6605849, upper bound: 0.6605157
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6600525, upper bound: 0.6613721
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6613721, upper bound: 0.6600525
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6605157, upper bound: 0.6605849
time: 0.30 seconds

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
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

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
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6605439
time: 0.30 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.04 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.04
Output dim: 0, lower bound: -0.6605439, upper bound: 0.6619682
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.04
Output dim: 0, lower bound: -0.6576264, upper bound: 0.6619682
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.04
Output dim: 0, lower bound: -0.6605849, upper bound: 0.6605157
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.04
Output dim: 0, lower bound: -0.6600525, upper bound: 0.6613721
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.04
Output dim: 0, lower bound: -0.6613721, upper bound: 0.6600525
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.04
Output dim: 0, lower bound: -0.6605157, upper bound: 0.6605849
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.04
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6576264
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.04
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6605439
Binary search (step 6): status=Status.VERIFIED, low=0.1804113, high=0.1818182, mid=0.1804113, abs_max=0.7819017171859741
rel_dist={0: [-0.7126414684186158, 0.7126414685280116]}

## Binary search (step 7) starts
Candidate diff: 0.1811147


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7053783, upper bound: 0.7051275
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7051275, upper bound: 0.7053783
time: 0.27 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.74 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.74
Output dim: 0, lower bound: -0.7053783, upper bound: 0.7051275
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.74
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
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7039676, upper bound: 0.6948314
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6905401, upper bound: 0.7037169
time: 0.29 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7037169, upper bound: 0.6905401
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6948314, upper bound: 0.7039676
time: 0.28 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.26 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 0, lower bound: -0.7039676, upper bound: 0.6948314
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 0, lower bound: -0.6905401, upper bound: 0.7037169
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 0, lower bound: -0.7037169, upper bound: 0.6905401
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 0, lower bound: -0.6948314, upper bound: 0.7039676

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6605439, upper bound: 0.6619682
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6576264, upper bound: 0.6619682
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6605849, upper bound: 0.6605157
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6600525, upper bound: 0.6613721
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6613721, upper bound: 0.6600525
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6605157, upper bound: 0.6605849
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

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
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6605439
time: 0.30 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.04 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.04
Output dim: 0, lower bound: -0.6605439, upper bound: 0.6619682
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.04
Output dim: 0, lower bound: -0.6576264, upper bound: 0.6619682
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.04
Output dim: 0, lower bound: -0.6605849, upper bound: 0.6605157
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.04
Output dim: 0, lower bound: -0.6600525, upper bound: 0.6613721
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.04
Output dim: 0, lower bound: -0.6613721, upper bound: 0.6600525
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.04
Output dim: 0, lower bound: -0.6605157, upper bound: 0.6605849
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.04
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6576264
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.04
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6605439
Binary search (step 7): status=Status.VERIFIED, low=0.1811147, high=0.1818182, mid=0.1811147, abs_max=0.7819017171859741
rel_dist={0: [-0.7126414684186158, 0.7126414685280116]}

## Binary search (step 8) starts
Candidate diff: 0.1814665


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

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
- Time for RS candidates: 0.72 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.72
Output dim: 0, lower bound: -0.7053783, upper bound: 0.7051275
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.72
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
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7039676, upper bound: 0.6948314
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6905401, upper bound: 0.7037169
time: 0.29 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

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
- Time for RS candidates: 2.27 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.27
Output dim: 0, lower bound: -0.7039676, upper bound: 0.6948314
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.27
Output dim: 0, lower bound: -0.6905401, upper bound: 0.7037169
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.27
Output dim: 0, lower bound: -0.7037169, upper bound: 0.6905401
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.27
Output dim: 0, lower bound: -0.6948314, upper bound: 0.7039676

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6605439, upper bound: 0.6619682
time: 0.32 seconds

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

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

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

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6613721, upper bound: 0.6600525
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6605157, upper bound: 0.6605849
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

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
- Time for RS candidates: 3.03 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.03
Output dim: 0, lower bound: -0.6605439, upper bound: 0.6619682
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.03
Output dim: 0, lower bound: -0.6576264, upper bound: 0.6619682
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.03
Output dim: 0, lower bound: -0.6605849, upper bound: 0.6605157
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.03
Output dim: 0, lower bound: -0.6600525, upper bound: 0.6613721
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.03
Output dim: 0, lower bound: -0.6613721, upper bound: 0.6600525
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.03
Output dim: 0, lower bound: -0.6605157, upper bound: 0.6605849
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.03
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6576264
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.03
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6605439
Binary search (step 8): status=Status.VERIFIED, low=0.1814665, high=0.1818182, mid=0.1814665, abs_max=0.7819017171859741
rel_dist={0: [-0.7126414684186158, 0.7126414685280116]}

## Binary search (step 9) starts
Candidate diff: 0.1816423


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

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
- Time for RS candidates: 0.72 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.72
Output dim: 0, lower bound: -0.7053783, upper bound: 0.7051275
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.72
Output dim: 0, lower bound: -0.7051275, upper bound: 0.7053783

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6905401, upper bound: 0.6948314
time: 0.30 seconds

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

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
Output dim: 0, lower bound: -0.6905401, upper bound: 0.7039676
time: 0.30 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.26 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 0, lower bound: -0.6905401, upper bound: 0.6948314
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 0, lower bound: -0.6905401, upper bound: 0.7037169
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 0, lower bound: -0.7037169, upper bound: 0.6905401
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 0, lower bound: -0.6905401, upper bound: 0.7039676

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6605439, upper bound: 0.6619682
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6576264, upper bound: 0.6619682
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6605849, upper bound: 0.6605157
time: 0.29 seconds

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

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

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
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

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
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6605439
time: 0.30 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.11 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.11
Output dim: 0, lower bound: -0.6605439, upper bound: 0.6619682
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.11
Output dim: 0, lower bound: -0.6576264, upper bound: 0.6619682
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.11
Output dim: 0, lower bound: -0.6605849, upper bound: 0.6605157
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.11
Output dim: 0, lower bound: -0.6600525, upper bound: 0.6613721
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.11
Output dim: 0, lower bound: -0.6613721, upper bound: 0.6600525
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.11
Output dim: 0, lower bound: -0.6605157, upper bound: 0.6605849
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.11
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6576264
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.11
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6605439
Binary search (step 9): status=Status.VERIFIED, low=0.1816423, high=0.1818182, mid=0.1816423, abs_max=0.7819017171859741
rel_dist={0: [-0.7126414684186158, 0.7126414685280116]}

## Binary search (step 10) starts
Candidate diff: 0.1817303


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7053783, upper bound: 0.7051275
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7051275, upper bound: 0.7053783
time: 0.28 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.72 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.72
Output dim: 0, lower bound: -0.7053783, upper bound: 0.7051275
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.72
Output dim: 0, lower bound: -0.7051275, upper bound: 0.7053783

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7039676, upper bound: 0.6948314
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6905401, upper bound: 0.7037169
time: 0.29 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6948314, upper bound: 0.6905401
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6948314, upper bound: 0.7039676
time: 0.30 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.24 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.24
Output dim: 0, lower bound: -0.7039676, upper bound: 0.6948314
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.24
Output dim: 0, lower bound: -0.6905401, upper bound: 0.7037169
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.24
Output dim: 0, lower bound: -0.6948314, upper bound: 0.6905401
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.24
Output dim: 0, lower bound: -0.6948314, upper bound: 0.7039676

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

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

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6605849, upper bound: 0.6605157
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6600525, upper bound: 0.6613721
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

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
time: 0.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

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
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6605439
time: 0.31 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.04 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.04
Output dim: 0, lower bound: -0.6605439, upper bound: 0.6619682
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.04
Output dim: 0, lower bound: -0.6576264, upper bound: 0.6619682
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.04
Output dim: 0, lower bound: -0.6605849, upper bound: 0.6605157
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.04
Output dim: 0, lower bound: -0.6600525, upper bound: 0.6613721
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.04
Output dim: 0, lower bound: -0.6613721, upper bound: 0.6600525
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.04
Output dim: 0, lower bound: -0.6605157, upper bound: 0.6605849
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.04
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6576264
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.04
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6605439
Binary search (step 10): status=Status.VERIFIED, low=0.1817303, high=0.1818182, mid=0.1817303, abs_max=0.7819017171859741
rel_dist={0: [-0.7126414684186158, 0.7126414685280116]}

## Binary search (step 11) starts
Candidate diff: 0.1817742


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

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
- Time for RS candidates: 0.71 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.71
Output dim: 0, lower bound: -0.7053783, upper bound: 0.7051275
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.71
Output dim: 0, lower bound: -0.7051275, upper bound: 0.7053783

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
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7037169, upper bound: 0.6948314
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6905401, upper bound: 0.7037169
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
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

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
time: 0.31 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.26 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 0, lower bound: -0.7037169, upper bound: 0.6948314
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 0, lower bound: -0.6905401, upper bound: 0.7037169
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 0, lower bound: -0.7037169, upper bound: 0.6905401
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.26
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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6605439, upper bound: 0.6619682
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6576264, upper bound: 0.6619682
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6605849, upper bound: 0.6605157
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6600525, upper bound: 0.6613721
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6613721, upper bound: 0.6600525
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6605157, upper bound: 0.6605849
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

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
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6605439
time: 0.30 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.05 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.05
Output dim: 0, lower bound: -0.6605439, upper bound: 0.6619682
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.05
Output dim: 0, lower bound: -0.6576264, upper bound: 0.6619682
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.05
Output dim: 0, lower bound: -0.6605849, upper bound: 0.6605157
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.05
Output dim: 0, lower bound: -0.6600525, upper bound: 0.6613721
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.05
Output dim: 0, lower bound: -0.6613721, upper bound: 0.6600525
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.05
Output dim: 0, lower bound: -0.6605157, upper bound: 0.6605849
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.05
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6576264
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.05
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6605439
Binary search (step 11): status=Status.VERIFIED, low=0.1817742, high=0.1818182, mid=0.1817742, abs_max=0.7819017171859741
rel_dist={0: [-0.7126414684186158, 0.7126414685280116]}

## Binary search (step 12) starts
Candidate diff: 0.1817962


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7053783, upper bound: 0.7051275
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7051275, upper bound: 0.7053783
time: 0.27 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.72 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.72
Output dim: 0, lower bound: -0.7053783, upper bound: 0.7051275
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.72
Output dim: 0, lower bound: -0.7051275, upper bound: 0.7053783

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7039676, upper bound: 0.6948314
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6905401, upper bound: 0.7037169
time: 0.29 seconds

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7037169, upper bound: 0.6905401
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6948314, upper bound: 0.7039676
time: 0.30 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.26 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 0, lower bound: -0.7039676, upper bound: 0.6948314
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 0, lower bound: -0.6905401, upper bound: 0.7037169
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 0, lower bound: -0.7037169, upper bound: 0.6905401
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 0, lower bound: -0.6948314, upper bound: 0.7039676

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

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

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6605849, upper bound: 0.6605157
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6600525, upper bound: 0.6613721
time: 0.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

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
Output dim: 0, lower bound: -0.6576264, upper bound: 0.6605849
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

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
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6605439
time: 0.30 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.06 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.06
Output dim: 0, lower bound: -0.6605439, upper bound: 0.6619682
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.06
Output dim: 0, lower bound: -0.6576264, upper bound: 0.6619682
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.06
Output dim: 0, lower bound: -0.6605849, upper bound: 0.6605157
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.06
Output dim: 0, lower bound: -0.6600525, upper bound: 0.6613721
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.06
Output dim: 0, lower bound: -0.6613721, upper bound: 0.6600525
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.06
Output dim: 0, lower bound: -0.6576264, upper bound: 0.6605849
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.06
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6576264
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.06
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6605439
Binary search (step 12): status=Status.VERIFIED, low=0.1817962, high=0.1818182, mid=0.1817962, abs_max=0.7819017171859741
rel_dist={0: [-0.7126414684186158, 0.7126414685280116]}

## Binary search (step 13) starts
Candidate diff: 0.1818072


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7053783, upper bound: 0.7051275
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7051275, upper bound: 0.7053783
time: 0.28 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.72 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.72
Output dim: 0, lower bound: -0.7053783, upper bound: 0.7051275
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.72
Output dim: 0, lower bound: -0.7051275, upper bound: 0.7053783

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7039676, upper bound: 0.6948314
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6905401, upper bound: 0.7037169
time: 0.29 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6948314, upper bound: 0.6905401
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6948314, upper bound: 0.7039676
time: 0.30 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.24 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.24
Output dim: 0, lower bound: -0.7039676, upper bound: 0.6948314
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.24
Output dim: 0, lower bound: -0.6905401, upper bound: 0.7037169
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.24
Output dim: 0, lower bound: -0.6948314, upper bound: 0.6905401
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.24
Output dim: 0, lower bound: -0.6948314, upper bound: 0.7039676

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6605439, upper bound: 0.6619682
time: 0.32 seconds

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

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6605849, upper bound: 0.6605157
time: 0.28 seconds

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

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6613721, upper bound: 0.6600525
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6605157, upper bound: 0.6605849
time: 0.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

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
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6605439
time: 0.29 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.01 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.01
Output dim: 0, lower bound: -0.6605439, upper bound: 0.6619682
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.01
Output dim: 0, lower bound: -0.6576264, upper bound: 0.6619682
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.01
Output dim: 0, lower bound: -0.6605849, upper bound: 0.6605157
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.01
Output dim: 0, lower bound: -0.6600525, upper bound: 0.6613721
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.01
Output dim: 0, lower bound: -0.6613721, upper bound: 0.6600525
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.01
Output dim: 0, lower bound: -0.6605157, upper bound: 0.6605849
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.01
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6576264
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.01
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6605439
Binary search (step 13): status=Status.VERIFIED, low=0.1818072, high=0.1818182, mid=0.1818072, abs_max=0.7819017171859741
rel_dist={0: [-0.7126414684186158, 0.7126414685280116]}

## Binary search (step 14) starts
Candidate diff: 0.1818127


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

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
time: 0.28 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.73 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.73
Output dim: 0, lower bound: -0.7053783, upper bound: 0.7051275
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.73
Output dim: 0, lower bound: -0.7051275, upper bound: 0.7053783

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6905401, upper bound: 0.6948314
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6905401, upper bound: 0.7037169
time: 0.29 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

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
time: 0.32 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.29 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.29
Output dim: 0, lower bound: -0.6905401, upper bound: 0.6948314
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.29
Output dim: 0, lower bound: -0.6905401, upper bound: 0.7037169
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.29
Output dim: 0, lower bound: -0.7037169, upper bound: 0.6905401
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.29
Output dim: 0, lower bound: -0.6948314, upper bound: 0.7039676

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6605439, upper bound: 0.6619682
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6576264, upper bound: 0.6619682
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

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
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6613721, upper bound: 0.6600525
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6605157, upper bound: 0.6605849
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

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
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6605439
time: 0.29 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.05 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.05
Output dim: 0, lower bound: -0.6605439, upper bound: 0.6619682
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.05
Output dim: 0, lower bound: -0.6576264, upper bound: 0.6619682
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.05
Output dim: 0, lower bound: -0.6605849, upper bound: 0.6605157
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.05
Output dim: 0, lower bound: -0.6600525, upper bound: 0.6613721
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.05
Output dim: 0, lower bound: -0.6613721, upper bound: 0.6600525
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.05
Output dim: 0, lower bound: -0.6605157, upper bound: 0.6605849
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.05
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6576264
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.05
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6605439
Binary search (step 14): status=Status.VERIFIED, low=0.1818127, high=0.1818182, mid=0.1818127, abs_max=0.7819017171859741
rel_dist={0: [-0.7126414684186158, 0.7126414685280116]}

## Binary search (step 15) starts
Candidate diff: 0.1818154


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

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
time: 0.28 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.73 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.73
Output dim: 0, lower bound: -0.7053783, upper bound: 0.7051275
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.73
Output dim: 0, lower bound: -0.7051275, upper bound: 0.7053783

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

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
time: 0.29 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6948314, upper bound: 0.6905401
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6948314, upper bound: 0.7039676
time: 0.30 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.29 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.29
Output dim: 0, lower bound: -0.7039676, upper bound: 0.6948314
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.29
Output dim: 0, lower bound: -0.6905401, upper bound: 0.7037169
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.29
Output dim: 0, lower bound: -0.6948314, upper bound: 0.6905401
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.29
Output dim: 0, lower bound: -0.6948314, upper bound: 0.7039676

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

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
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

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

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

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
time: 0.29 seconds

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
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

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
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6605439
time: 0.29 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.05 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.05
Output dim: 0, lower bound: -0.6605439, upper bound: 0.6619682
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.05
Output dim: 0, lower bound: -0.6576264, upper bound: 0.6619682
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.05
Output dim: 0, lower bound: -0.6605849, upper bound: 0.6605157
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.05
Output dim: 0, lower bound: -0.6600525, upper bound: 0.6613721
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.05
Output dim: 0, lower bound: -0.6613721, upper bound: 0.6600525
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.05
Output dim: 0, lower bound: -0.6605157, upper bound: 0.6605849
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.05
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6576264
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.05
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6605439
Binary search (step 15): status=Status.VERIFIED, low=0.1818154, high=0.1818182, mid=0.1818154, abs_max=0.7819017171859741
rel_dist={0: [-0.7126414684186158, 0.7126414685280116]}

## Binary search (step 16) starts
Candidate diff: 0.1818168


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

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
- Time for RS candidates: 0.73 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.73
Output dim: 0, lower bound: -0.7053783, upper bound: 0.7051275
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.73
Output dim: 0, lower bound: -0.7051275, upper bound: 0.7053783

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7039676, upper bound: 0.6948314
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6905401, upper bound: 0.7037169
time: 0.29 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6948314, upper bound: 0.6905401
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6948314, upper bound: 0.7039676
time: 0.30 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.33 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.33
Output dim: 0, lower bound: -0.7039676, upper bound: 0.6948314
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.33
Output dim: 0, lower bound: -0.6905401, upper bound: 0.7037169
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.33
Output dim: 0, lower bound: -0.6948314, upper bound: 0.6905401
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.33
Output dim: 0, lower bound: -0.6948314, upper bound: 0.7039676

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6605439, upper bound: 0.6619682
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6576264, upper bound: 0.6619682
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6605849, upper bound: 0.6605157
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6600525, upper bound: 0.6613721
time: 0.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

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
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

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
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6605439
time: 0.29 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.02 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.02
Output dim: 0, lower bound: -0.6605439, upper bound: 0.6619682
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.02
Output dim: 0, lower bound: -0.6576264, upper bound: 0.6619682
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.02
Output dim: 0, lower bound: -0.6605849, upper bound: 0.6605157
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.02
Output dim: 0, lower bound: -0.6600525, upper bound: 0.6613721
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.02
Output dim: 0, lower bound: -0.6613721, upper bound: 0.6600525
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.02
Output dim: 0, lower bound: -0.6605157, upper bound: 0.6605849
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.02
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6576264
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.02
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6605439
Binary search (step 16): status=Status.VERIFIED, low=0.1818168, high=0.1818182, mid=0.1818168, abs_max=0.7819017171859741
rel_dist={0: [-0.7126414684186158, 0.7126414684186158]}

## Binary search (step 17) starts
Candidate diff: 0.1818175


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

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
- Time for RS candidates: 0.72 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.72
Output dim: 0, lower bound: -0.7053783, upper bound: 0.7051275
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.72
Output dim: 0, lower bound: -0.7051275, upper bound: 0.7053783

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7039676, upper bound: 0.6948314
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6905401, upper bound: 0.7037169
time: 0.29 seconds

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7037169, upper bound: 0.6905401
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6948314, upper bound: 0.7039676
time: 0.30 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.27 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.27
Output dim: 0, lower bound: -0.7039676, upper bound: 0.6948314
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.27
Output dim: 0, lower bound: -0.6905401, upper bound: 0.7037169
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.27
Output dim: 0, lower bound: -0.7037169, upper bound: 0.6905401
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.27
Output dim: 0, lower bound: -0.6948314, upper bound: 0.7039676

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6605439, upper bound: 0.6619682
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6576264, upper bound: 0.6619682
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

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
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6613721, upper bound: 0.6600525
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6605157, upper bound: 0.6605849
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0323386, 0.7495631, -0.0323386, 0.7495631, -0.7819017, 0.7819017
1: -0.1552227, 0.9255341, -0.1552227, 0.9255341, -1.0807568, 1.0807568
2: -0.0741289, 0.9267865, -0.0741289, 0.9267865, -1.0009153, 1.0009153
3: -0.3143692, 0.9027434, -0.3143692, 0.9027434, -1.2171125, 1.2171125
4: -0.2515757, 0.9410419, -0.2515757, 0.9410419, -1.1926177, 1.1926177

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

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
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6605439
time: 0.30 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.21 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.21
Output dim: 0, lower bound: -0.6605439, upper bound: 0.6619682
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.21
Output dim: 0, lower bound: -0.6576264, upper bound: 0.6619682
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.21
Output dim: 0, lower bound: -0.6605849, upper bound: 0.6605157
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.21
Output dim: 0, lower bound: -0.6600525, upper bound: 0.6613721
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.21
Output dim: 0, lower bound: -0.6613721, upper bound: 0.6600525
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.21
Output dim: 0, lower bound: -0.6605157, upper bound: 0.6605849
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.21
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6576264
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.21
Output dim: 0, lower bound: -0.6619682, upper bound: 0.6605439
Binary search (step 17): status=Status.VERIFIED, low=0.1818175, high=0.1818182, mid=0.1818175, abs_max=0.7819017171859741
rel_dist={0: [-0.7126414684186158, 0.7126414685280116]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.1818175002593681
execution time: 360.99 seconds
